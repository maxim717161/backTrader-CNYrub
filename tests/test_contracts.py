from datetime import date, timedelta

from cnyrub.contracts import (
    Contract,
    candidate_secids,
    contracts_document,
    discover_contracts,
    front_windows,
    parse_contract,
)


def contract(secid: str, shortname: str, frst: str, lst: str) -> Contract:
    first = date.fromisoformat(frst)
    last = date.fromisoformat(lst)
    return Contract(secid, shortname, "CNY", first, last, last, 1000)


def fields(**overrides: str | None) -> dict[str, str | None]:
    payload: dict[str, str | None] = {
        "SECID": "CRZ5",
        "SHORTNAME": "CNY-12.25",
        "ASSETCODE": "CNY",
        "TYPE": "futures",
        "LOTSIZE": "1000",
        "FRSTTRADE": "2024-06-06",
        "LSTTRADE": "2025-12-18",
        "LSTDELDATE": "2025-12-18",
    }
    payload.update(overrides)
    return payload


SCHEDULE = [
    contract("CRM2", "CNY-6.22", "2022-04-21", "2022-06-16"),
    contract("CRU2", "CNY-9.22", "2022-04-21", "2022-09-15"),
    contract("CRZ2", "CNY-12.22", "2022-06-02", "2022-12-15"),
    contract("CRH3", "CNY-3.23", "2022-08-15", "2023-03-16"),
    contract("CRM3", "CNY-6.23", "2022-09-02", "2023-06-15"),
    contract("CRU3", "CNY-9.23", "2022-11-02", "2023-09-21"),
    contract("CRZ3", "CNY-12.23", "2022-11-02", "2023-12-21"),
    contract("CRH4", "CNY-3.24", "2022-11-02", "2024-03-21"),
    contract("CRM4", "CNY-6.24", "2022-12-02", "2024-06-20"),
    contract("CRU4", "CNY-9.24", "2023-03-02", "2024-09-19"),
    contract("CRZ4", "CNY-12.24", "2023-06-01", "2024-12-19"),
    contract("CRH5", "CNY-3.25", "2023-09-08", "2025-03-20"),
    contract("CRM5", "CNY-6.25", "2023-12-08", "2025-06-19"),
    contract("CRU5", "CNY-9.25", "2024-03-07", "2025-09-18"),
    contract("CRZ5", "CNY-12.25", "2024-06-06", "2025-12-18"),
    contract("CRH6", "CNY-3.26", "2024-09-06", "2026-03-19"),
    contract("CRM6", "CNY-6.26", "2024-12-06", "2026-06-18"),
    contract("CRU6", "CNY-9.26", "2025-03-07", "2026-09-17"),
    contract("CRZ6", "CNY-12.26", "2025-06-05", "2026-12-17"),
    contract("CRH7", "CNY-3.27", "2025-09-10", "2027-03-18"),
    contract("CRM7", "CNY-6.27", "2025-12-05", "2027-06-17"),
    contract("CRU7", "CNY-9.27", "2026-03-06", "2027-09-16"),
    contract("CRZ7", "CNY-12.27", "2026-06-04", "2027-12-16"),
    contract("CRH8", "CNY-3.28", "2026-09-04", "2028-03-16"),
]

AS_OF = date(2026, 9, 21)

EXPECTED_WINDOWS = [
    ("CRM2", "2022-04-21", "2022-06-16"),
    ("CRU2", "2022-06-17", "2022-09-15"),
    ("CRZ2", "2022-09-16", "2022-12-15"),
    ("CRH3", "2022-12-16", "2023-03-16"),
    ("CRM3", "2023-03-17", "2023-06-15"),
    ("CRU3", "2023-06-16", "2023-09-21"),
    ("CRZ3", "2023-09-22", "2023-12-21"),
    ("CRH4", "2023-12-22", "2024-03-21"),
    ("CRM4", "2024-03-22", "2024-06-20"),
    ("CRU4", "2024-06-21", "2024-09-19"),
    ("CRZ4", "2024-09-20", "2024-12-19"),
    ("CRH5", "2024-12-20", "2025-03-20"),
    ("CRM5", "2025-03-21", "2025-06-19"),
    ("CRU5", "2025-06-20", "2025-09-18"),
    ("CRZ5", "2025-09-19", "2025-12-18"),
    ("CRH6", "2025-12-19", "2026-03-19"),
    ("CRM6", "2026-03-20", "2026-06-18"),
    ("CRU6", "2026-06-19", "2026-09-17"),
    ("CRZ6", "2026-09-18", "2026-09-21"),
]


def test_parse_keeps_quarterly_cny_and_drops_other_instruments():
    parsed = parse_contract(fields())
    assert parsed is not None
    assert parsed.secid == "CRZ5"
    assert parsed.lotsize == 1000
    assert parse_contract(fields(SECID="CNYRUBF", SHORTNAME="CNYRUBF")) is None
    assert parse_contract(fields(SECID="CRH7CRM7", SHORTNAME="CNY-3.27-6.27")) is None
    assert parse_contract(fields(SECID="UCZ6", SHORTNAME="UCNY-12.26")) is None
    assert parse_contract(fields(SECID="MYZ6", SHORTNAME="MOEXCNY-12.26")) is None
    assert parse_contract(fields(SECID="CRZ5", SHORTNAME="CNY-12.26")) is None
    assert parse_contract(fields(ASSETCODE="UCNY")) is None
    assert parse_contract(fields(LOTSIZE="1")) is None


def test_discover_asks_iss_and_ignores_empty_codes():
    seen: list[str] = []

    def fetch(secid: str) -> dict[str, str | None] | None:
        seen.append(secid)
        if secid == "CRM2":
            return fields(
                SECID="CRM2",
                SHORTNAME="CNY-6.22",
                FRSTTRADE="2022-04-21",
                LSTTRADE="2022-06-16",
                LSTDELDATE="2022-06-16",
            )
        return None

    found = discover_contracts(date(2026, 9, 21), fetch)
    assert [item.secid for item in found] == ["CRM2"]
    assert "CNYRUBF" not in seen
    assert "CRM2" in seen
    assert len(seen) == len(set(seen))


def test_candidate_codes_cover_listed_quarters_without_repeating_a_digit():
    secids = candidate_secids(date(2026, 9, 21))
    assert secids[0] == "CRH2"
    assert "CRZ6" in secids
    assert "CRH8" in secids
    assert len(secids) == len(set(secids))


def test_front_windows_on_21_september_2026():
    windows = front_windows(SCHEDULE, AS_OF)
    assert [(item.secid, item.start.isoformat(), item.end.isoformat()) for item in windows] == EXPECTED_WINDOWS
    assert windows[0].start == SCHEDULE[0].frsttrade
    for previous, current in zip(windows, windows[1:]):
        assert current.start == previous.contract.lsttrade + timedelta(days=1)
        assert previous.end == previous.contract.lsttrade


def test_second_contract_starts_after_previous_expiry_not_on_its_listing_date():
    early_listing = contract("CRU2", "CNY-9.22", "2022-03-01", "2022-09-15")
    first = contract("CRM2", "CNY-6.22", "2022-04-21", "2022-06-16")
    windows = front_windows([early_listing, first], date(2022, 8, 1))
    assert [(item.secid, item.start, item.end) for item in windows] == [
        ("CRM2", date(2022, 4, 21), date(2022, 6, 16)),
        ("CRU2", date(2022, 6, 17), date(2022, 8, 1)),
    ]


def test_expiration_day_still_belongs_to_the_expiring_contract():
    windows = front_windows(SCHEDULE, date(2026, 9, 17))
    assert windows[-1].secid == "CRU6"
    assert windows[-1].end == date(2026, 9, 17)
    assert "CRZ6" not in {item.secid for item in windows}


def test_document_marks_far_quarters():
    document = contracts_document(SCHEDULE, AS_OF)
    assert document["front"] == "CRZ6"
    by_id = {row["secid"]: row for row in document["contracts"]}  # type: ignore[index]
    assert by_id["CRZ6"]["in_series"] is True
    assert by_id["CRZ6"]["window_end"] == "2026-09-21"
    assert by_id["CRH7"]["in_series"] is False
    assert by_id["CRH7"]["window_start"] is None
    assert "CNYRUBF" not in by_id
