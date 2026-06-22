"""One-shot KIS mock-investment runner using only Python standard library.

Use this when `requests` cannot be installed. It reads `.env`, issues an access
token, queries current price and balance, then prints the finance strategy
decision. If `DRY_RUN=false`, it submits the selected mock order.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from account import AccountSnapshot
from config import SEOUL_TZ, Settings
from strategy import decide_action


BASE_URL = "https://openapivts.koreainvestment.com:29443"
NEW_YORK_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Candidate:
    symbol: str
    name: str
    market: str
    exchange_code: str = "KRX"


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    current_price: int
    snapshot: AccountSnapshot
    decision: object


def parse_candidates(raw: str, market: str, default_exchange: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        symbol = parts[0]
        name = parts[1] if len(parts) >= 2 and parts[1] else symbol
        exchange = parts[2] if len(parts) >= 3 and parts[2] else default_exchange
        candidates.append(Candidate(symbol=symbol, name=name, market=market, exchange_code=exchange))
    return candidates


def active_market(now: Optional[datetime] = None) -> str:
    now_kst = now.astimezone(SEOUL_TZ) if now else datetime.now(SEOUL_TZ)
    now_ny = now_kst.astimezone(NEW_YORK_TZ)

    if now_kst.weekday() < 5 and dt_time(9, 0) <= now_kst.time() < dt_time(15, 30):
        return "kr"
    if now_ny.weekday() < 5 and dt_time(9, 30) <= now_ny.time() < dt_time(16, 0):
        return "us"
    return "closed"


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
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc
    payload = json.loads(raw)
    if payload.get("rt_cd") not in (None, "0"):
        raise RuntimeError(f"KIS API error {payload.get('msg_cd')}: {payload.get('msg1')}")
    time.sleep(0.5)
    return payload


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
    return str(token)


def headers(token: str, app_key: str, app_secret: str, tr_id: str) -> dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "tr_id": tr_id,
        "custtype": "P",
    }


def get_price(token: str, app_key: str, app_secret: str, symbol: str) -> int:
    payload = request_json(
        "GET",
        f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price",
        headers=headers(token, app_key, app_secret, "FHKST01010100"),
        params={
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
        },
    )
    return int(payload["output"]["stck_prpr"])


def get_us_price(token: str, app_key: str, app_secret: str, symbol: str, exchange_code: str) -> int:
    payload = request_json(
        "GET",
        f"{BASE_URL}/uapi/overseas-price/v1/quotations/price",
        headers=headers(token, app_key, app_secret, "HHDFS00000300"),
        params={
            "AUTH": "",
            "EXCD": exchange_code,
            "SYMB": symbol,
        },
    )
    output = payload.get("output") or {}
    raw_price = output.get("last") or output.get("base") or output.get("ovrs_nmix_prpr")
    return int(float(str(raw_price).replace(",", "")))


def get_balance(
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
    product_code: str,
    symbol: str,
) -> AccountSnapshot:
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
    holdings = payload.get("output1") or []

    def to_int(value) -> int:
        try:
            return int(float(str(value or "0").replace(",", "")))
        except ValueError:
            return 0

    rows = [row for row in holdings if str(row.get("pdno", "")).strip() == symbol]
    quantity = sum(to_int(row.get("hldg_qty")) for row in rows)
    avg = 0
    if rows:
        avg = to_int(rows[0].get("pchs_avg_pric") or rows[0].get("pchs_avg_price"))
    output2 = payload.get("output2") or {}
    if isinstance(output2, list):
        output2 = output2[0] if output2 else {}
    cash = to_int(output2.get("dnca_tot_amt") or output2.get("prvs_rcdl_excc_amt") or output2.get("nass_amt"))
    return AccountSnapshot(symbol_quantity=quantity, available_cash=cash, average_price=avg)


def score_result(result: CandidateResult) -> tuple[int, float, float]:
    decision = result.decision
    action_score = {"buy": 2, "hold": 1, "sell": 0}.get(getattr(decision, "action", "hold"), 1)
    return (
        action_score,
        float(getattr(decision, "expected_return", 0.0)),
        -float(getattr(decision, "loss_probability", 1.0)),
    )


def choose_best(results: list[CandidateResult]) -> CandidateResult:
    buyable = [item for item in results if getattr(item.decision, "action", "") == "buy"]
    if buyable:
        return max(buyable, key=score_result)
    return max(results, key=score_result)


def run_candidate_scan(
    *,
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
    product_code: str,
    settings: Settings,
    candidates: list[Candidate],
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for candidate in candidates:
        if candidate.market == "kr":
            price = get_price(token, app_key, app_secret, candidate.symbol)
            snapshot = get_balance(
                token,
                app_key,
                app_secret,
                account_no,
                product_code,
                candidate.symbol,
            )
            cash = snapshot.available_cash
        else:
            price = get_us_price(token, app_key, app_secret, candidate.symbol, candidate.exchange_code)
            # Overseas balance/order APIs use separate KIS endpoints. Keep this as
            # signal-only unless ENABLE_US_TRADING is explicitly implemented/enabled.
            cash = int(os.getenv("US_SIGNAL_CASH", "10000"))
            snapshot = AccountSnapshot(symbol_quantity=0, available_cash=cash, average_price=0)

        candidate_settings = Settings(
            app_key=app_key,
            app_secret=app_secret,
            account_number=account_no,
            account_product_code=product_code,
            symbol=candidate.symbol,
            market_code="J" if candidate.market == "kr" else candidate.exchange_code,
            exchange_code="KRX" if candidate.market == "kr" else candidate.exchange_code,
            price_offset_krw=settings.price_offset_krw,
            order_quantity=settings.order_quantity,
            dry_run=settings.dry_run,
            order_division=settings.order_division,
            strategy_mode=settings.strategy_mode,
            annual_drift=settings.annual_drift,
            annual_volatility=settings.annual_volatility,
            simulation_days=settings.simulation_days,
            simulation_paths=settings.simulation_paths,
            min_expected_return=settings.min_expected_return,
            max_loss_probability=settings.max_loss_probability,
            take_profit_pct=settings.take_profit_pct,
            stop_loss_pct=settings.stop_loss_pct,
        )
        decision = decide_action(current_price=price, snapshot=snapshot, settings=candidate_settings)
        results.append(CandidateResult(candidate, price, snapshot, decision))
    return results


def submit_order(
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
    product_code: str,
    symbol: str,
    side: str,
    quantity: int,
    price: int,
    order_division: str,
):
    tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"
    return request_json(
        "POST",
        f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash",
        headers=headers(token, app_key, app_secret, tr_id),
        body={
            "CANO": account_no,
            "ACNT_PRDT_CD": product_code,
            "PDNO": symbol,
            "ORD_DVSN": order_division,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
        },
    )


def main() -> None:
    load_env(Path(".env"))
    account_no, product_code = split_account(os.environ["GH_ACCOUNT"])
    app_key = os.environ["GH_APPKEY"]
    app_secret = os.environ["GH_APPSECRET"]
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    enable_us_trading = os.getenv("ENABLE_US_TRADING", "false").lower() == "true"
    market = os.getenv("FORCE_MARKET", "auto").strip().lower()
    if market == "auto":
        market = active_market()

    settings = Settings.from_env()
    kr_candidates = parse_candidates(
        os.getenv("KR_CANDIDATES", "069500:KODEX200,005930:삼성전자,000660:SK하이닉스"),
        "kr",
        "KRX",
    )
    us_candidates = parse_candidates(
        os.getenv("US_CANDIDATES", "TQQQ:TQQQ:NAS,NVDA:엔비디아:NAS,SOXL:SOXL:NAS"),
        "us",
        "NAS",
    )
    candidates = kr_candidates if market == "kr" else us_candidates if market == "us" else []

    token = issue_token(app_key, app_secret)
    print("token_issued: true")

    if not candidates:
        print("market_status: closed")
        print("No scan/order submitted")
        status = {
            "timestamp": datetime.now(SEOUL_TZ).isoformat(timespec="seconds"),
            "market": market,
            "dry_run": dry_run,
            "action": "hold",
            "reason": "Neither Korea nor US regular market is open",
        }
        logs = Path("logs")
        logs.mkdir(exist_ok=True)
        Path("logs/status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        with Path("logs/trader.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(status, ensure_ascii=False) + "\n")
        return

    print(f"active_market: {market}")
    print("candidates:", ", ".join(f"{item.symbol}({item.name})" for item in candidates))
    results = run_candidate_scan(
        token=token,
        app_key=app_key,
        app_secret=app_secret,
        account_no=account_no,
        product_code=product_code,
        settings=settings,
        candidates=candidates,
    )
    for result in results:
        decision = result.decision
        print(
            "scan:",
            f"{result.candidate.symbol}({result.candidate.name})",
            f"price={result.current_price:,}",
            f"action={decision.action}",
            f"expected_return={decision.expected_return:.4%}",
            f"loss_probability={decision.loss_probability:.2%}",
            f"reason={decision.reason}",
        )

    best = choose_best(results)
    symbol = best.candidate.symbol
    price = best.current_price
    snapshot = best.snapshot
    decision = best.decision
    print(f"selected: {symbol}({best.candidate.name})")
    print(
        "decision:",
        f"action={decision.action}",
        f"buy_price={decision.buy_price:,}",
        f"sell_price={decision.sell_price:,}",
        f"expected_return={decision.expected_return:.4%}",
        f"loss_probability={decision.loss_probability:.2%}",
        f"reason={decision.reason}",
    )
    if best.candidate.market == "us" and decision.action in {"buy", "sell"} and not enable_us_trading:
        print("US signal only: ENABLE_US_TRADING=false, order was not submitted")
        order_payload = {
            "dry_run": True,
            "message": "US signal only. Overseas order endpoint is not enabled.",
        }
    elif best.candidate.market == "us" and decision.action in {"buy", "sell"}:
        print("US order skipped: overseas order submission is not implemented in this stdlib runner")
        order_payload = {
            "message": "US overseas order submission is not implemented in this runner",
        }
    elif decision.action in {"buy", "sell"} and not dry_run:
        order_price = decision.buy_price if decision.action == "buy" else decision.sell_price
        payload = submit_order(
            token,
            app_key,
            app_secret,
            account_no,
            product_code,
            symbol,
            decision.action,
            settings.order_quantity,
            order_price,
            settings.order_division,
        )
        print("order_response:", json.dumps(payload, ensure_ascii=False, indent=2))
        order_payload = payload
    elif decision.action in {"buy", "sell"}:
        print("DRY_RUN=true: order was not submitted")
        order_payload = {"dry_run": True, "message": "order was not submitted"}
    else:
        print("No order submitted")
        order_payload = {"message": "no order submitted"}

    status = {
        "timestamp": datetime.now(SEOUL_TZ).isoformat(timespec="seconds"),
        "market": market,
        "symbol": symbol,
        "name": best.candidate.name,
        "dry_run": dry_run,
        "current_price": price,
        "quantity": snapshot.symbol_quantity,
        "cash": snapshot.available_cash,
        "average_price": snapshot.average_price,
        "action": decision.action,
        "buy_price": decision.buy_price,
        "sell_price": decision.sell_price,
        "expected_return": decision.expected_return,
        "loss_probability": decision.loss_probability,
        "reason": decision.reason,
        "order": order_payload,
        "scan": [
            {
                "symbol": item.candidate.symbol,
                "name": item.candidate.name,
                "market": item.candidate.market,
                "current_price": item.current_price,
                "action": item.decision.action,
                "expected_return": item.decision.expected_return,
                "loss_probability": item.decision.loss_probability,
                "reason": item.decision.reason,
            }
            for item in results
        ],
    }
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    Path("logs/status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path("logs/trader.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(status, ensure_ascii=False) + "\n")
    print("status_saved: logs/status.json")


if __name__ == "__main__":
    main()
