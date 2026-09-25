from datetime import date

from cnyrub.iss import PAGE_SIZE, candles_request, collect_pages


def test_candles_url_matches_the_public_iss_path():
    url, params = candles_request("CRZ5", date(2025, 9, 19), date(2025, 12, 18), 500)
    assert url == (
        "https://iss.moex.com/iss/engines/futures/markets/forts/boards/RFUD/securities/CRZ5/candles.json"
    )
    assert params == {
        "interval": 1,
        "from": "2025-09-19",
        "till": "2025-12-18",
        "start": 500,
        "iss.meta": "off",
    }


def test_collect_pages_follows_offsets_until_a_short_page():
    pages = {
        0: [[0]] * PAGE_SIZE,
        PAGE_SIZE: [[1]] * PAGE_SIZE,
        PAGE_SIZE * 2: [[2]] * 3,
    }

    def fetch(offset: int) -> list[list[int]]:
        return pages.get(offset, [])

    rows = collect_pages(fetch)
    assert len(rows) == PAGE_SIZE * 2 + 3


def test_collect_pages_asks_for_the_page_after_an_exact_batch():
    calls: list[int] = []

    def fetch(offset: int) -> list[list[int]]:
        calls.append(offset)
        if offset == 0:
            return [[0]] * PAGE_SIZE
        return []

    rows = collect_pages(fetch)
    assert calls == [0, PAGE_SIZE]
    assert len(rows) == PAGE_SIZE
