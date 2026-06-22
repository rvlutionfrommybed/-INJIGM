from orders import MOCK_ORDER_TR_IDS


def test_mock_order_transaction_ids() -> None:
    assert MOCK_ORDER_TR_IDS["buy"] == "VTTC0802U"
    assert MOCK_ORDER_TR_IDS["sell"] == "VTTC0801U"
