from config import _split_account


def test_split_account_with_hyphen() -> None:
    assert _split_account("12345678-01") == ("12345678", "01")


def test_split_account_without_hyphen() -> None:
    assert _split_account("1234567801") == ("12345678", "01")

