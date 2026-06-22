"""Session-aware KIS mock trader for Korean and US watchlists.

Korean regular session scans KODEX 200, Samsung Electronics, and SK Hynix.
US regular session scans TQQQ, NVIDIA, and SOXL. The script selects the active
market by clock time, ranks candidates by Brownian expected return minus a
loss-probability penalty, and submits at most one mock limit order when
DRY_RUN=false.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from config import SEOUL_TZ
from simulation import simulate_gbm_paths


BASE_URL = "https://openapivts.koreainvestment.com:29443"
NEW_YORK_TZ = ZoneInfo("America/New_York")
TOKEN_CACHE = Path("token_cache.json")


@dataclass(frozen=True)
class Asset:
    market: str
    symbol: str
    name: str
    exchange: str
    currency: str
    quantity: int


@dataclass(frozen=True)
class CandidateResult:
    asset: Asset
    price: float
    buy_price: float
    sell_price: float
    expected_return: float
    loss_probability: float
    score: float
    action: str
    reason: str


def request_interval() -> float:
    return float(os.getenv("REQUEST_INTERVAL_SECONDS", "1.2"))


def retry_delay() -> float:
    return float(os.getenv("RATE_LIMIT_RETRY_SECONDS", "2.5"))


KOREA_WATCHLIST = [
    Asset("KR", "069500", "KODEX 200", "KRX", "KRW", 1),
    Asset("KR", "005930", "Samsung Electronics", "KRX", "KRW", 1),
    Asset("KR", "000660", "SK Hynix", "KRX", "KRW", 1),
]

US_WATCHLIST = [
    Asset("US", "TQQQ", "ProShares UltraPro QQQ", "NASD", "USD", 1),
    Asset("US", "NVDA", "NVIDIA", "NASD", "USD", 1),
    Asset("US", "SOXL", "Direxion Daily Semiconductor Bull 3X", "AMEX", "USD", 1),
]


def load_env(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing .env file: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def split_account(account: str) -> tuple[str, str]:
    normalized = account.replace("-", "").strip()
    if len(normalized) != 10 or not normalized.isdigit():
        raise ValueError("GH_ACCOUNT must look like 12345678-01")
    return normalized[:8], normalized[8:]


def request_json(method: str, url: str, headers=None, params=None, body=None):
    headers = headers or {}
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    attempts = int(os.getenv("RATE_LIMIT_RETRIES", "3"))
    for attempt in range(attempts + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            if "EGW00201" in raw and attempt < attempts:
                time.sleep(retry_delay() * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc
        payload = json.loads(raw)
        if payload.get("rt_cd") not in (None, "0"):
            if payload.get("msg_cd") == "EGW00201" and attempt < attempts:
                time.sleep(retry_delay() * (attempt + 1))
                continue
            raise RuntimeError(f"KIS API error {payload.get('msg_cd')}: {payload.get('msg1')}")
        time.sleep(request_interval())
        return payload
    raise RuntimeError("KIS request failed after rate-limit retries")


def issue_token(app_key: str, app_secret: str) -> str:
    payload = request_json(
        "POST",
        f"{BASE_URL}/oauth2/tokenP",
        headers={"content-type": "application/json; charset=utf-8"},
        body={
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret,
        },
    )
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"Token response missing access_token: {payload}")
    expires_in = int(payload.get("expires_in", 0) or 0)
    cache = {
        "access_token": token,
        "expires_at": time.time() + max(expires_in - 300, 60),
    }
    TOKEN_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(token)


def get_token(app_key: str, app_secret: str) -> str:
    if TOKEN_CACHE.exists():
        try:
            cache = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
            if cache.get("access_token") and float(cache.get("expires_at", 0)) > time.time():
                return str(cache["access_token"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return issue_token(app_key, app_secret)


def headers(token: str, app_key: str, app_secret: str, tr_id: str) -> dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "tr_id": tr_id,
        "custtype": "P",
    }


def to_float(value) -> float:
    try:
        return float(str(value or "0").replace(",", ""))
    except ValueError:
        return 0.0


def to_int(value) -> int:
    return int(to_float(value))


def active_watchlist(now_seoul: datetime) -> tuple[str, list[Asset]]:
    force_market = os.getenv("FORCE_MARKET", "").strip().upper()
    if force_market == "KR":
        return "KR", KOREA_WATCHLIST
    if force_market == "US":
        return "US", US_WATCHLIST
    if os.getenv("TRADING_WINDOW_MODE", "regular").strip().lower() == "always":
        return "KR", KOREA_WATCHLIST

    now_ny = now_seoul.astimezone(NEW_YORK_TZ)
    korea_open = now_seoul.weekday() < 5 and dtime(9, 0) <= now_seoul.time() < dtime(15, 30)
    us_open = now_ny.weekday() < 5 and dtime(9, 30) <= now_ny.time() < dtime(16, 0)
    if korea_open:
        return "KR", KOREA_WATCHLIST
    if us_open:
        return "US", US_WATCHLIST
    return "CLOSED", []


def get_domestic_price(token: str, app_key: str, app_secret: str, symbol: str) -> float:
    payload = request_json(
        "GET",
        f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price",
        headers=headers(token, app_key, app_secret, "FHKST01010100"),
        params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
    )
    return to_float(payload["output"]["stck_prpr"])


def get_overseas_price(token: str, app_key: str, app_secret: str, asset: Asset) -> float:
    payload = request_json(
        "GET",
        f"{BASE_URL}/uapi/overseas-price/v1/quotations/price",
        headers=headers(token, app_key, app_secret, "HHDFS00000300"),
        params={"AUTH": "", "EXCD": asset.exchange, "SYMB": asset.symbol},
    )
    output = payload.get("output") or {}
    price = to_float(output.get("last") or output.get("stck_prpr") or output.get("base"))
    if price <= 0:
        raise RuntimeError(f"Unexpected overseas price response for {asset.symbol}: {output}")
    return price


def domestic_available_cash(
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
    product_code: str,
) -> float:
    payload = request_json(
        "GET",
        f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance",
        headers=headers(token, app_key, app_secret, "VTTC8434R"),
        params={
            "CANO": account_no,
            "ACNT_PRDT_CD": product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )
    output2 = payload.get("output2") or {}
    if isinstance(output2, list):
        output2 = output2[0] if output2 else {}
    return to_float(output2.get("dnca_tot_amt") or output2.get("prvs_rcdl_excc_amt") or output2.get("nass_amt"))


def scan_candidate(asset: Asset, price: float) -> CandidateResult:
    drift = float(os.getenv("ANNUAL_DRIFT", "0.05"))
    volatility = float(os.getenv("ANNUAL_VOLATILITY", "0.20"))
    days = int(os.getenv("SIMULATION_DAYS", "20"))
    paths_count = int(os.getenv("SIMULATION_PATHS", "500"))
    min_expected_return = float(os.getenv("MIN_EXPECTED_RETURN", "0.001"))
    max_loss_probability = float(os.getenv("MAX_LOSS_PROBABILITY", "0.48"))
    loss_penalty = float(os.getenv("LOSS_PROBABILITY_PENALTY", "0.5"))
    limit_offset_pct = float(os.getenv("LIMIT_OFFSET_PCT", "0.003"))

    paths = simulate_gbm_paths(
        spot=price,
        drift=drift,
        volatility=volatility,
        days=days,
        paths=paths_count,
        seed=sum(ord(ch) for ch in asset.symbol),
    )
    terminal_prices = [path[-1] for path in paths]
    expected_terminal = statistics.mean(terminal_prices)
    expected_return = expected_terminal / price - 1.0
    loss_probability = sum(item < price for item in terminal_prices) / len(terminal_prices)
    score = expected_return - loss_penalty * loss_probability

    decimals = 0 if asset.currency == "KRW" else 2
    buy_price = round(price * (1.0 - limit_offset_pct), decimals)
    sell_price = round(price * (1.0 + limit_offset_pct), decimals)
    if asset.currency == "KRW":
        buy_price = math.floor(buy_price / 5) * 5
        sell_price = math.ceil(sell_price / 5) * 5

    if expected_return >= min_expected_return and loss_probability <= max_loss_probability:
        action = "buy"
        reason = "Expected return and loss probability passed the risk filter"
    else:
        action = "hold"
        reason = "Risk filter failed"

    return CandidateResult(
        asset=asset,
        price=price,
        buy_price=buy_price,
        sell_price=sell_price,
        expected_return=expected_return,
        loss_probability=loss_probability,
        score=score,
        action=action,
        reason=reason,
    )


def submit_domestic_order(
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
    product_code: str,
    result: CandidateResult,
):
    return request_json(
        "POST",
        f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash",
        headers=headers(token, app_key, app_secret, "VTTC0802U"),
        body={
            "CANO": account_no,
            "ACNT_PRDT_CD": product_code,
            "PDNO": result.asset.symbol,
            "ORD_DVSN": os.getenv("ORDER_DIVISION", "00"),
            "ORD_QTY": str(result.asset.quantity),
            "ORD_UNPR": str(int(result.buy_price)),
        },
    )


def submit_overseas_order(
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
    product_code: str,
    result: CandidateResult,
):
    return request_json(
        "POST",
        f"{BASE_URL}/uapi/overseas-stock/v1/trading/order",
        headers=headers(token, app_key, app_secret, "VTTT1002U"),
        body={
            "CANO": account_no,
            "ACNT_PRDT_CD": product_code,
            "OVRS_EXCG_CD": result.asset.exchange,
            "PDNO": result.asset.symbol,
            "ORD_DVSN": os.getenv("OVERSEAS_ORDER_DIVISION", "00"),
            "ORD_QTY": str(result.asset.quantity),
            "OVRS_ORD_UNPR": f"{result.buy_price:.2f}",
            "ORD_SVR_DVSN_CD": "0",
        },
    )


def result_to_dict(result: CandidateResult) -> dict[str, object]:
    return {
        "market": result.asset.market,
        "symbol": result.asset.symbol,
        "name": result.asset.name,
        "exchange": result.asset.exchange,
        "currency": result.asset.currency,
        "price": result.price,
        "buy_price": result.buy_price,
        "sell_price": result.sell_price,
        "expected_return": result.expected_return,
        "loss_probability": result.loss_probability,
        "score": result.score,
        "action": result.action,
        "reason": result.reason,
    }


def main() -> None:
    load_env(Path(".env"))
    account_no, product_code = split_account(os.environ["GH_ACCOUNT"])
    app_key = os.environ["GH_APPKEY"]
    app_secret = os.environ["GH_APPSECRET"]
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    now_seoul = datetime.now(SEOUL_TZ)
    session, watchlist = active_watchlist(now_seoul)

    if not watchlist:
        print("session: closed")
        print("No active Korean or US regular session. Use FORCE_MARKET=KR or FORCE_MARKET=US for testing.")
        return

    token = get_token(app_key, app_secret)
    print("token_ready: true")
    print(f"session: {session}")

    results: list[CandidateResult] = []
    for asset in watchlist:
        try:
            price = (
                get_domestic_price(token, app_key, app_secret, asset.symbol)
                if asset.market == "KR"
                else get_overseas_price(token, app_key, app_secret, asset)
            )
            result = scan_candidate(asset, price)
        except RuntimeError as exc:
            print(f"{asset.symbol:>6} {asset.name:<36} skipped: {exc}")
            continue
        results.append(result)
        print(
            f"{asset.symbol:>6} {asset.name:<36}",
            f"price={price:,.2f}",
            f"expected={result.expected_return:.4%}",
            f"loss={result.loss_probability:.2%}",
            f"score={result.score:.4f}",
            f"action={result.action}",
        )

    if not results:
        raise RuntimeError("All candidates failed. Wait a minute and run again.")

    tradable = [item for item in results if item.action == "buy"]
    selected = max(tradable, key=lambda item: item.score) if tradable else max(results, key=lambda item: item.score)
    order_payload: dict[str, object]
    if selected.action != "buy":
        print(f"selected: hold ({selected.asset.symbol}) - no candidate passed the risk filter")
        order_payload = {"message": "no order submitted"}
    elif dry_run:
        print(f"selected: buy {selected.asset.symbol} at {selected.buy_price} ({selected.asset.currency})")
        print("DRY_RUN=true: order was not submitted")
        order_payload = {"dry_run": True, "message": "order was not submitted"}
    else:
        if selected.asset.market == "KR":
            cash = domestic_available_cash(token, app_key, app_secret, account_no, product_code)
            required_cash = selected.buy_price * selected.asset.quantity
            if cash < required_cash:
                print(f"selected: hold ({selected.asset.symbol}) - insufficient KRW cash")
                order_payload = {"message": "insufficient cash", "cash": cash, "required_cash": required_cash}
            else:
                try:
                    order_payload = submit_domestic_order(token, app_key, app_secret, account_no, product_code, selected)
                    print("order_response:", json.dumps(order_payload, ensure_ascii=False, indent=2))
                except RuntimeError as exc:
                    print(f"order_rejected: {exc}")
                    order_payload = {
                        "accepted": False,
                        "message": str(exc),
                        "selected_symbol": selected.asset.symbol,
                    }
        elif os.getenv("ENABLE_US_TRADING", "false").strip().lower() != "true":
            print("selected: US signal only - ENABLE_US_TRADING=false")
            order_payload = {
                "dry_run": True,
                "message": "US candidate selected, but overseas order submission is disabled",
            }
        else:
            try:
                order_payload = submit_overseas_order(token, app_key, app_secret, account_no, product_code, selected)
                print("order_response:", json.dumps(order_payload, ensure_ascii=False, indent=2))
            except RuntimeError as exc:
                print(f"order_rejected: {exc}")
                order_payload = {
                    "accepted": False,
                    "message": str(exc),
                    "selected_symbol": selected.asset.symbol,
                }

    status = {
        "timestamp": now_seoul.isoformat(timespec="seconds"),
        "session": session,
        "dry_run": dry_run,
        "selected": result_to_dict(selected),
        "candidates": [result_to_dict(item) for item in results],
        "order": order_payload,
    }
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    Path("logs/status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path("logs/trader.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(status, ensure_ascii=False) + "\n")
    print("status_saved: logs/status.json")


if __name__ == "__main__":
    main()
