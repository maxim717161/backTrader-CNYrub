"""Квартальные фьючерсы CNY/RUB и их фронтальные окна."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

# Торговля квартальным CNY/RUB началась в апреле 2022. Код контракта — CR,
# буква месяца и последняя цифра года, поэтому окно поиска держим короче
# десяти лет: иначе новый выпуск перезапишет старый с той же цифрой года.
FIRST_YEAR = 2022
MONTH_LETTERS = ("H", "M", "U", "Z")
LETTER_TO_MONTH = {"H": 3, "M": 6, "U": 9, "Z": 12}

_SECID = re.compile(r"^CR([HMUZ])(\d)$")
_SHORTNAME = re.compile(r"^CNY-(3|6|9|12)\.(\d{2})$")


@dataclass(frozen=True)
class Contract:
    secid: str
    shortname: str
    assetcode: str
    frsttrade: date
    lsttrade: date
    lstdeldate: date | None
    lotsize: int


@dataclass(frozen=True)
class Window:
    contract: Contract
    start: date
    end: date

    @property
    def secid(self) -> str:
        return self.contract.secid


def candidate_secids(today: date) -> list[str]:
    """Коды, которые могли быть выпущены с 2022 года по сегодня плюс три года."""
    last_year = today.year + 3
    secids: list[str] = []
    seen: set[str] = set()
    for year in range(FIRST_YEAR, last_year + 1):
        for letter in MONTH_LETTERS:
            secid = f"CR{letter}{year % 10}"
            if secid in seen:
                continue
            seen.add(secid)
            secids.append(secid)
    return secids


def parse_contract(fields: dict[str, str | None]) -> Contract | None:
    """Оставить квартальный фьючерс на курс юань-рубль с лотом 1000.

    Вечный CNYRUBF, календарные спреды, UCNY и MOEXCNY сюда не проходят:
    у них другой код или другое краткое имя.
    """
    secid = fields.get("SECID") or ""
    secid_match = _SECID.match(secid)
    shortname = fields.get("SHORTNAME") or ""
    short_match = _SHORTNAME.match(shortname)
    if secid_match is None or short_match is None:
        return None
    letter, year_digit = secid_match.group(1), int(secid_match.group(2))
    month, year_suffix = int(short_match.group(1)), int(short_match.group(2))
    if LETTER_TO_MONTH[letter] != month or year_suffix % 10 != year_digit:
        return None
    if fields.get("ASSETCODE") != "CNY":
        return None
    instrument_type = fields.get("TYPE")
    if instrument_type not in (None, "futures"):
        return None
    lot_raw = fields.get("LOTSIZE") or "1000"
    try:
        lotsize = int(float(lot_raw))
    except (TypeError, ValueError):
        return None
    if lotsize != 1000:
        return None
    frst_raw = fields.get("FRSTTRADE")
    lst_raw = fields.get("LSTTRADE")
    if not frst_raw or not lst_raw:
        return None
    lstdel_raw = fields.get("LSTDELDATE") or None
    try:
        frsttrade = date.fromisoformat(frst_raw)
        lsttrade = date.fromisoformat(lst_raw)
        lstdeldate = date.fromisoformat(lstdel_raw) if lstdel_raw else None
    except ValueError:
        return None
    if lsttrade < frsttrade:
        return None
    return Contract(
        secid=secid,
        shortname=shortname,
        assetcode="CNY",
        frsttrade=frsttrade,
        lsttrade=lsttrade,
        lstdeldate=lstdeldate,
        lotsize=lotsize,
    )


def discover_contracts(
    today: date,
    fetch_description: Callable[[str], dict[str, str | None] | None],
) -> list[Contract]:
    """Собрать выпуски, которые ISS реально отдаёт по коду контракта."""
    found: dict[str, Contract] = {}
    for secid in candidate_secids(today):
        fields = fetch_description(secid)
        if not fields:
            continue
        contract = parse_contract(fields)
        if contract is not None:
            found[contract.secid] = contract
    return sorted(found.values(), key=lambda contract: (contract.lsttrade, contract.secid))


def front_windows(contracts: list[Contract], today: date) -> list[Window]:
    """Окна, на которых контракт был ближайшим к экспирации.

    Начало первого выпуска — его первый день торгов. У остальных начало —
    следующий календарный день после последнего дня торгов предыдущего.
    Конец закрытого контракта — его последний день торгов. У того, который
    ещё торгуется, конец — день загрузки. Дальние кварталы не входят.
    """
    ordered = sorted(contracts, key=lambda contract: (contract.lsttrade, contract.secid))
    if not ordered:
        return []
    still_trading = [contract for contract in ordered if contract.lsttrade >= today]
    if still_trading:
        front = still_trading[0]
        included = [contract for contract in ordered if contract.lsttrade <= front.lsttrade]
    else:
        included = list(ordered)

    windows: list[Window] = []
    previous_last: date | None = None
    for contract in included:
        start = contract.frsttrade if previous_last is None else previous_last + timedelta(days=1)
        end = today if contract.lsttrade >= today else contract.lsttrade
        if start <= end:
            windows.append(Window(contract=contract, start=start, end=end))
        previous_last = contract.lsttrade
    return windows


def contracts_document(contracts: list[Contract], today: date) -> dict[str, object]:
    """Описание всех найденных выпусков и фронтальных окон на дату."""
    windows = {window.secid: window for window in front_windows(contracts, today)}
    front_secid = next(
        (window.secid for window in windows.values() if window.contract.lsttrade >= today),
        None,
    )
    rows = []
    for contract in sorted(contracts, key=lambda item: (item.lsttrade, item.secid)):
        window = windows.get(contract.secid)
        rows.append(
            {
                "secid": contract.secid,
                "shortname": contract.shortname,
                "assetcode": contract.assetcode,
                "frsttrade": contract.frsttrade.isoformat(),
                "lsttrade": contract.lsttrade.isoformat(),
                "lstdeldate": contract.lstdeldate.isoformat() if contract.lstdeldate else None,
                "lotsize": contract.lotsize,
                "in_series": window is not None,
                "window_start": window.start.isoformat() if window else None,
                "window_end": window.end.isoformat() if window else None,
            }
        )
    return {"as_of": today.isoformat(), "front": front_secid, "contracts": rows}
