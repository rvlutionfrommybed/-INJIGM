"""One-month daily backtest for the multimarket finance strategy.

This script uses KIS domestic daily candles for the Korean watchlist and runs a
simple walk-forward test:

1. Estimate drift/volatility from only prior closes.
2. Rank candidates with the same expected-return/loss-probability idea.
3. Buy the best passing candidate at today's close.
4. Mark the position at the next trading day's close.

It is intentionally daily-close based, so it validates the signal logic rather
than claiming exact intraday fill quality.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from config import SEOUL_TZ
from multi_market_live_once_stdlib import (
    BASE_URL,
    KOREA_WATCHLIST,
    get_token,
    headers,
    load_env,
    request_json,
    split_account,
    to_float,
)
from simulation import simulate_gbm_paths


@dataclass(frozen=True)
class DailyClose:
    date: str
    close: float


@dataclass(frozen=True)
class SignalRow:
    date: str
    symbol: str
    name: str
    close: float
    next_close: float
    expected_return: float
    loss_probability: float
    score: float
    realized_return: float
    equity: float


def date_yyyymmdd(days_ago: int) -> str:
    return (datetime.now(SEOUL_TZ) - timedelta(days=days_ago)).strftime("%Y%m%d")


def fetch_daily_closes(token: str, app_key: str, app_secret: str, symbol: str) -> list[DailyClose]:
    payload = request_json(
        "GET",
        f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        headers=headers(token, app_key, app_secret, "FHKST03010100"),
        params={
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": date_yyyymmdd(75),
            "FID_INPUT_DATE_2": date_yyyymmdd(0),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "1",
        },
    )
    rows = []
    for item in payload.get("output2") or []:
        date = str(item.get("stck_bsop_date", "")).strip()
        close = to_float(item.get("stck_clpr"))
        if date and close > 0:
            rows.append(DailyClose(date=date, close=close))
    rows.sort(key=lambda item: item.date)
    return rows


def estimate_annual_params(closes: list[float]) -> tuple[float, float]:
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(returns) < 2:
        return 0.05, 0.20
    drift = statistics.mean(returns) * 252.0
    volatility = statistics.stdev(returns) * math.sqrt(252.0)
    return drift, max(volatility, 0.0001)


def score_candidate(symbol: str, close: float, prior_closes: list[float]) -> tuple[float, float, float]:
    drift, volatility = estimate_annual_params(prior_closes)
    paths = simulate_gbm_paths(
        spot=close,
        drift=drift,
        volatility=volatility,
        days=int(os.getenv("BACKTEST_SIMULATION_DAYS", "5")),
        paths=int(os.getenv("SIMULATION_PATHS", "500")),
        seed=sum(ord(ch) for ch in symbol),
    )
    terminal = [path[-1] for path in paths]
    expected_return = statistics.mean(terminal) / close - 1.0
    loss_probability = sum(price < close for price in terminal) / len(terminal)
    score = expected_return - float(os.getenv("LOSS_PROBABILITY_PENALTY", "0.5")) * loss_probability
    return expected_return, loss_probability, score


def run_backtest(history: dict[str, list[DailyClose]]) -> list[SignalRow]:
    lookback = int(os.getenv("BACKTEST_LOOKBACK_DAYS", "10"))
    min_expected = float(os.getenv("MIN_EXPECTED_RETURN", "0.001"))
    max_loss_probability = float(os.getenv("MAX_LOSS_PROBABILITY", "0.48"))
    initial_cash = float(os.getenv("BACKTEST_INITIAL_CASH", "10000000"))
    equity = initial_cash

    common_dates = sorted(set.intersection(*[set(row.date for row in rows) for rows in history.values()]))
    rows_by_symbol = {symbol: {row.date: row.close for row in rows} for symbol, rows in history.items()}
    results: list[SignalRow] = []

    for idx in range(lookback, len(common_dates) - 1):
        date = common_dates[idx]
        next_date = common_dates[idx + 1]
        ranked = []
        for asset in KOREA_WATCHLIST:
            closes = [rows_by_symbol[asset.symbol][day] for day in common_dates[idx - lookback : idx + 1]]
            close = closes[-1]
            expected_return, loss_probability, score = score_candidate(asset.symbol, close, closes[:-1])
            if expected_return >= min_expected and loss_probability <= max_loss_probability:
                ranked.append((score, expected_return, loss_probability, asset, close))

        if not ranked:
            continue

        score, expected_return, loss_probability, asset, close = max(ranked, key=lambda item: item[0])
        next_close = rows_by_symbol[asset.symbol][next_date]
        realized_return = next_close / close - 1.0
        equity *= 1.0 + realized_return
        results.append(
            SignalRow(
                date=date,
                symbol=asset.symbol,
                name=asset.name,
                close=close,
                next_close=next_close,
                expected_return=expected_return,
                loss_probability=loss_probability,
                score=score,
                realized_return=realized_return,
                equity=equity,
            )
        )
    return results[-22:]


def write_outputs(rows: list[SignalRow]) -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "backtest_last_month.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SignalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    if rows:
        total_return = rows[-1].equity / float(os.getenv("BACKTEST_INITIAL_CASH", "10000000")) - 1.0
        win_rate = sum(row.realized_return > 0 for row in rows) / len(rows)
        avg_daily = statistics.mean(row.realized_return for row in rows)
    else:
        total_return = win_rate = avg_daily = 0.0

    report = [
        "# Last-Month Backtest",
        "",
        f"- Trades: {len(rows)}",
        f"- Total return: {total_return:.4%}",
        f"- Win rate: {win_rate:.2%}",
        f"- Average next-day return: {avg_daily:.4%}",
        "",
        "This is a daily-close signal backtest, not an intraday fill simulation.",
    ]
    (out_dir / "backtest_last_month.md").write_text("\n".join(report), encoding="utf-8")
    print(f"saved: {csv_path}")
    print("saved: outputs/backtest_last_month.md")
    print("\n".join(report))


def main() -> None:
    load_env(Path(".env"))
    split_account(os.environ["GH_ACCOUNT"])
    app_key = os.environ["GH_APPKEY"]
    app_secret = os.environ["GH_APPSECRET"]
    token = get_token(app_key, app_secret)
    print("token_ready: true")

    history = {}
    for asset in KOREA_WATCHLIST:
        rows = fetch_daily_closes(token, app_key, app_secret, asset.symbol)
        history[asset.symbol] = rows
        print(f"history: {asset.symbol} {asset.name} rows={len(rows)}")
        time.sleep(float(os.getenv("REQUEST_INTERVAL_SECONDS", "1.3")))

    backtest_rows = run_backtest(history)
    write_outputs(backtest_rows)


if __name__ == "__main__":
    main()
