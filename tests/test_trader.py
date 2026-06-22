import logging
from pathlib import Path

from account import AccountSnapshot
from config import Settings
from trader import PendingOrder, Trader


def _settings() -> Settings:
    return Settings(
        app_key="key",
        app_secret="secret",
        account_number="12345678",
        account_product_code="01",
        token_cache_path=Path("token_cache.json"),
        log_dir=Path("logs"),
    )


def test_pending_buy_is_cleared_when_quantity_increases() -> None:
    trader = Trader(_settings(), logging.getLogger("test"))
    trader.pending["buy"] = PendingOrder("buy", 0)
    trader._clear_filled_pending(AccountSnapshot(symbol_quantity=1, available_cash=0))
    assert "buy" not in trader.pending

