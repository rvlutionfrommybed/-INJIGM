from datetime import datetime, timedelta

from auth import TokenManager
from config import SEOUL_TZ


def test_same_day_unexpired_token_is_reusable() -> None:
    now = datetime.now(SEOUL_TZ)
    payload = {
        "access_token": "test",
        "issued_date": now.date().isoformat(),
        "expires_at": (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    assert TokenManager._is_reusable(payload)


def test_old_token_is_not_reusable() -> None:
    payload = {
        "access_token": "test",
        "issued_date": "2000-01-01",
        "expires_at": "2000-01-01 23:59:59",
    }
    assert not TokenManager._is_reusable(payload)

