"""Кворум чужих позиций и выбор правила для суммы восьми счетов."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from consensus import OpenState, StrategyStats
from cnyrub.engine import _FillBook
from trend import (
    SPECS,
    START_SUM,
    TrendSpec,
    _MomentumTape,
    apply_momentum_target,
    _roles_decision,
    choose_roles,
    choose_spec,
    combine_veto,
    choose_fortyfive,
    choose_momentum,
    choose_thirty,
    entry_veto,
    fortyfive_spec,
    kept_split,
    thirty_spec,
    make_split,
    m1760_spec,
    momentum_spec,
    sixty_spec,
    RoleRule,
    plot_kept,
    obvious_sources,
    obvious_veto,
    plot_equity,
    plot_roles,
    run_eight,
    write_roles_curve,
)


def _state(name: str, channel: int, side: int, *, held_days: int = 2, pnl: float = 1_000.0) -> OpenState:
    return OpenState(name, channel, side, held_days, pnl, pnl, None, 0.0)


def _spec(**overrides: object) -> TrendSpec:
    data: dict[str, object] = dict(id="probe", title="probe", enabled=True, quorum=2, hold_days=1)
    data.update(overrides)
    return TrendSpec(**data)  # type: ignore[arg-type]


def test_two_longer_positions_block_the_opposite_entry_and_one_does_not():
    peers = [
        _state("m1760", 1760, -1),
        _state("m2160", 2160, -1),
        _state("thirty", 30, -1),
    ]
    spec = _spec()
    assert obvious_veto(peers, spec, "short", 525, 0) == 1
    assert obvious_sources(peers, spec, "short", 525, 1, 0) == ["m1760", "m2160"]
    assert obvious_veto(peers[:1], spec, "short", 525, 0) == 0
    assert obvious_veto(peers, spec, "long", 12_420, 0) == 0


def test_disagreement_and_a_shield_leave_the_entry_open():
    mixed = [_state("m1760", 1760, -1), _state("m2160", 2160, 1)]
    spec = _spec()
    assert obvious_veto(mixed, spec, "short", 525, 0) == 0
    shielded = _spec(shield=frozenset({"short"}))
    peers = [_state("m1760", 1760, -1), _state("long", 12_420, -1)]
    assert obvious_veto(peers, shielded, "short", 525, 0) == 0


def test_momentum_votes_without_a_channel_and_a_short_window_does_not_block_it():
    peers = [_state("momentum", 0, -1), _state("long", 12_420, -1)]
    assert obvious_veto(peers, _spec(), "short", 525, 0) == 1
    assert obvious_veto(peers, _spec(momentum_votes=False), "short", 525, 0) == 0
    voters = [_state("thirty", 30, 1), _state("fortyfive", 45, 1), _state("long", 12_420, 1)]
    assert obvious_veto(voters, _spec(min_channel=1760), "momentum", 0, 0) == 0
    assert obvious_veto(voters, _spec(min_channel=1760), "momentum", 0, 0) == 0
    assert obvious_veto(
        [_state("m1760", 1760, 1), _state("long", 12_420, 1)],
        _spec(min_channel=1760),
        "momentum",
        0,
        0,
    ) == -1


def test_price_move_has_to_agree_with_the_quorum():
    peers = [_state("m1760", 1760, -1), _state("long", 12_420, -1)]
    spec = _spec(price_days=10)
    assert obvious_veto(peers, spec, "short", 525, -1) == 1
    assert obvious_veto(peers, spec, "short", 525, 1) == 0
    assert obvious_veto(peers, spec, "short", 525, 0) == 0


def test_legacy_veto_stacks_with_the_quorum():
    assert combine_veto(1, 0) == 1
    assert combine_veto(1, -1) == 2
    assert combine_veto(0, 0) == 0


def test_momentum_flattens_instead_of_flipping_into_a_veto():
    book = _FillBook("CRH4", 100_000.0)
    book.held = 4
    book.target = 4
    assert apply_momentum_target(book, -6, veto=1) is False
    assert book.target == -6
    book.held = 4
    assert apply_momentum_target(book, -6, veto=-1) is True
    assert book.target == 0
    flat = _FillBook("CRH4", 100_000.0)
    assert apply_momentum_target(flat, 5, veto=1) is True
    assert flat.target == 0
    assert apply_momentum_target(flat, 5, veto=-1) is False
    assert flat.target == 5


def test_weekend_minutes_do_not_open_a_momentum_session():
    tape = _MomentumTape()
    friday = date(2024, 1, 5)
    assert tape.see(friday, 10, 10.2, 9.8, 10.1, 5) is True
    assert tape.see(friday, 10.1, 10.3, 10.0, 10.2, 1) is False
    assert tape.see(date(2024, 1, 6), 10, 10, 10, 10, 1) is False
    assert tape.see(date(2024, 1, 8), 10.2, 10.4, 10.1, 10.3, 2) is True
    assert tape.closes == [10.2]
    assert tape.price_sign(1) == 0


def test_choice_keeps_the_long_window_and_then_the_sum():
    def item(spec_id: str, long_net: float, other: float, part: str = "CRM2") -> object:
        spec = next(spec for spec in SPECS if spec.id == spec_id) if spec_id in {"base", "legacy"} else _spec(id=spec_id, title=spec_id)
        strategies = [
            StrategyStats("long", [{"pnlcomm": long_net, "secid": part}], 0, 0, 0, 100_000 + long_net, -1, 100_000),
            StrategyStats("short", [{"pnlcomm": other, "secid": part}], 0, 0, 0, 100_000 + other, -1, 100_000),
        ]
        from trend import EightResult

        return EightResult(spec, strategies, -10, [], [])

    base = item("base", 100, 50)
    hurts = item("hurts", 90, 500)
    calm = item("calm", 100, 80, "CRH6")
    rich = item("rich", 110, 70)
    chosen = choose_spec([base, item("legacy", 100, 60), hurts, calm, rich])  # type: ignore[list-item]
    assert chosen.spec.id == "rich"
    # У calm часть CRM2 пустая, у базы она 150: calm слабее в этой части, rich нет.
    assert chosen is not calm


def test_plot_writes_three_lines(tmp_path: Path):
    from trend import EightResult

    def result(spec_id: str, level: float) -> EightResult:
        spec = TrendSpec(spec_id, spec_id, legacy=spec_id == "legacy")
        stats = StrategyStats("long", [], 0, 0, 0, level, 0, 100_000)
        return EightResult(spec, [stats], 0, [date(2024, 1, 2), date(2024, 1, 3)], [level, level + 10])

    path = tmp_path / "equity.png"
    results = [result("base", 800_000), result("legacy", 810_000), result("q2-d1", 820_000)]
    plot_equity(results, path)
    assert path.stat().st_size > 1000


def test_role_chart_labels_the_split(tmp_path: Path):
    from trend import EightResult

    def result(spec: object, level: float) -> EightResult:
        stats = StrategyStats("long", [], 0, 0, 0, level, 0, 100_000)
        return EightResult(spec, [stats], 0, [date(2024, 1, 2), date(2024, 1, 3)], [level, level + 10])  # type: ignore[arg-type]

    base = next(spec for spec in SPECS if spec.id == "base")
    uniform = next(spec for spec in SPECS if spec.id == "q4-d1")
    anchor = make_split("none", "none")
    results = [result(base, 800_000), result(uniform, 810_000), result(anchor, 820_000)]
    chart = tmp_path / "roles.png"
    table = tmp_path / "roles.csv"
    plot_roles(results, chart)
    write_roles_curve(results, table)
    assert chart.stat().st_size > 1000
    assert "uniform_q4" in table.read_text(encoding="utf-8")


def test_a_medium_rule_that_taxes_1760_stays_out_of_the_choice():
    from trend import EightResult

    names = ("short", "long", "thirty", "fortyfive", "sixty", "m1760", "m2160", "momentum")

    def book(spec: object, nets: dict[str, float]) -> EightResult:
        strategies = [
            StrategyStats(name, [{"pnlcomm": nets[name], "secid": "CRM2"}], 0, 0, 0, 100_000 + nets[name], -1, 100_000)
            for name in names
        ]
        return EightResult(spec, strategies, -10, [], [])  # type: ignore[arg-type]

    flat = {name: 1_000.0 for name in names}
    taxed = dict(flat)
    taxed["m1760"] = 900.0
    taxed["m2160"] = 1_200.0
    base = next(spec for spec in SPECS if spec.id == "base")
    legacy = next(spec for spec in SPECS if spec.id == "legacy")
    uniform = next(spec for spec in SPECS if spec.id == "q4-d1")
    anchor = make_split("none", "none")
    results = [
        book(base, flat),
        book(legacy, flat),
        book(uniform, flat),
        book(anchor, flat),
        book(make_split("mom-d2", "none"), taxed),
    ]
    assert choose_roles(results).spec.id == "roles-fixed"
    text = _roles_decision(*results[:4], results[3], results)
    assert "только моментум, 2 дня" in text
    assert "m1760" in text


def test_sixty_choice_refuses_a_gain_that_taxes_the_other_minis():
    from trend import EightResult, choose_sixty

    names = ("short", "long", "thirty", "fortyfive", "sixty", "m1760", "m2160", "momentum")

    def book(spec: object, nets: dict[str, float]) -> EightResult:
        strategies = [
            StrategyStats(name, [{"pnlcomm": nets[name], "secid": "CRM2"}], 0, 0, 0, 100_000 + nets[name], -1, 100_000)
            for name in names
        ]
        return EightResult(spec, strategies, -10, [], [])  # type: ignore[arg-type]

    flat = {name: 1_000.0 for name in names}
    taxed = dict(flat)
    taxed["sixty"] = 1_500.0
    taxed["fortyfive"] = 700.0
    safe = dict(flat)
    safe["sixty"] = 1_200.0
    base = next(spec for spec in SPECS if spec.id == "base")
    anchor = sixty_spec("leg", "старшее, 2 дня", RoleRule("legacy"))
    rich = sixty_spec("d1", "старшее, 1 день", RoleRule("quorum", quorum=1, hold_days=1, momentum_votes=False))
    costly = sixty_spec("none", "само", RoleRule("none"))
    results = [book(base, flat), book(anchor, flat), book(rich, safe), book(costly, taxed)]
    assert choose_sixty(results).spec.id == "sixty-d1"


def test_m1760_rule_does_not_move_the_other_roles():
    spec = m1760_spec(
        "long-d2",
        "только длинное",
        RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False, voters=frozenset({"long"})),
    )
    one_long = [_state("long", 12_420, -1, held_days=2)]
    wide = [_state("m2160", 2160, -1, held_days=2)]
    fresh = [_state("long", 12_420, -1, held_days=0)]
    momentum = [_state("momentum", 0, -1, held_days=2)]
    assert entry_veto("m1760", 1760, one_long, one_long, spec, 0) == 1
    assert entry_veto("m1760", 1760, wide, wide, spec, 0) == 0
    assert entry_veto("m1760", 1760, momentum, [], spec, 0) == 0
    assert entry_veto("sixty", 60, fresh, fresh, spec, 0) == 1
    assert entry_veto("thirty", 30, fresh, fresh, spec, 0) == 0
    assert entry_veto("m2160", 2160, momentum, [], spec, 0) == 1
    alone = m1760_spec("none", "само", RoleRule("none"))
    assert entry_veto("m1760", 1760, one_long, one_long, alone, 0) == 0


def test_m1760_choice_refuses_a_gain_that_taxes_the_short_window():
    from trend import EightResult, choose_m1760

    names = ("short", "long", "thirty", "fortyfive", "sixty", "m1760", "m2160", "momentum")

    def book(spec: object, nets: dict[str, float]) -> EightResult:
        strategies = [
            StrategyStats(name, [{"pnlcomm": nets[name], "secid": "CRM2"}], 0, 0, 0, 100_000 + nets[name], -1, 100_000)
            for name in names
        ]
        return EightResult(spec, strategies, -10, [], [])  # type: ignore[arg-type]

    flat = {name: 1_000.0 for name in names}
    taxed = dict(flat)
    taxed["m1760"] = 2_000.0
    taxed["short"] = 500.0
    safe = dict(flat)
    safe["m1760"] = 1_400.0
    base = next(spec for spec in SPECS if spec.id == "base")
    anchor = m1760_spec("none", "само", RoleRule("none"))
    rich = m1760_spec("long-d2", "только длинное", RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False))
    costly = m1760_spec("leg", "старшее", RoleRule("legacy"))
    results = [book(base, flat), book(anchor, flat), book(rich, safe), book(costly, taxed)]
    assert choose_m1760(results).spec.id == "m1760-long-d2"


def test_kept_book_blocks_sixty_as_soon_as_a_longer_window_is_in_profit():
    spec = kept_split()
    fresh = [_state("long", 12_420, -1, held_days=0)]
    ripe = [_state("long", 12_420, -1, held_days=2)]
    assert entry_veto("sixty", 60, fresh, fresh, spec, 0) == 1
    assert entry_veto("thirty", 30, fresh, fresh, spec, 0) == 0
    assert entry_veto("fortyfive", 45, ripe, ripe, spec, 0) == 0
    wide = [_state("m1760", 1760, -1, held_days=2)]
    fresh_wide = [_state("m1760", 1760, -1, held_days=1)]
    assert entry_veto("fortyfive", 45, wide, wide, spec, 0) == 1
    assert entry_veto("fortyfive", 45, fresh_wide, fresh_wide, spec, 0) == 0
    assert entry_veto("thirty", 30, wide, wide, spec, 0) == 1
    losing = [_state("m1760", 1760, -1, held_days=0, pnl=-100.0)]
    assert entry_veto("sixty", 60, losing, losing, spec, 0) == 0


def test_sixty_rule_stays_on_sixty_and_leaves_the_other_roles():
    alone = sixty_spec("none", "само", RoleRule("none"))
    momentum = [_state("momentum", 0, -1, held_days=2)]
    one_long = [_state("long", 12_420, -1, held_days=2)]
    assert entry_veto("sixty", 60, one_long, one_long, alone, 0) == 0
    assert entry_veto("thirty", 30, one_long, one_long, alone, 0) == 1
    assert entry_veto("m2160", 2160, momentum, [], alone, 0) == 1
    senior = [_state("sixty", 60, -1, held_days=2)]
    assert entry_veto("fortyfive", 45, senior, senior, alone, 0) == 0
    assert entry_veto("fortyfive", 45, [_state("m1760", 1760, -1, held_days=2)], [_state("m1760", 1760, -1, held_days=2)], alone, 0) == 1
    current = sixty_spec("leg", "старшее, 2 дня", RoleRule("legacy"))
    assert entry_veto("sixty", 60, one_long, one_long, current, 0) == 1
    only_long = sixty_spec(
        "long-d2",
        "только длинное",
        RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False, voters=frozenset({"long"})),
    )
    short_only = [_state("short", 525, -1, held_days=2)]
    assert entry_veto("sixty", 60, one_long, one_long, only_long, 0) == 1
    assert entry_veto("sixty", 60, short_only, short_only, only_long, 0) == 0
    assert entry_veto("thirty", 30, short_only, short_only, only_long, 0) == 1


def test_only_2160_yields_when_momentum_is_two_days_in_profit():
    spec = kept_split()
    fresh = [_state("momentum", 0, -1, held_days=1)]
    ripe = [_state("momentum", 0, -1, held_days=2)]
    assert entry_veto("m2160", 2160, ripe, [], spec, 0) == 1
    assert entry_veto("m2160", 2160, fresh, [], spec, 0) == 0
    assert entry_veto("m1760", 1760, ripe, [], spec, 0) == 0
    assert entry_veto("long", 12_420, ripe, [], spec, 0) == 0
    assert entry_veto("momentum", 0, ripe, [], spec, 0) == 0
    one_long = [_state("long", 12_420, -1, held_days=2)]
    assert entry_veto("thirty", 30, one_long, one_long, spec, 0) == 1
    assert entry_veto("short", 525, one_long, one_long, spec, 0) == 0


def test_kept_chart_shows_the_2160_gap(tmp_path: Path):
    from trend import EightResult, write_kept_curve

    def result(spec: object, level: float) -> EightResult:
        stats = StrategyStats("long", [], 0, 0, 0, level, 0, 100_000)
        return EightResult(spec, [stats], 0, [date(2024, 1, 2), date(2024, 1, 3)], [level, level + 10])  # type: ignore[arg-type]

    base = next(spec for spec in SPECS if spec.id == "base")
    results = [
        result(base, 800_000),
        result(make_split("none", "none"), 810_000),
        result(kept_split(), 820_000),
    ]
    chart = tmp_path / "kept.png"
    table = tmp_path / "kept.csv"
    plot_kept(results, chart)
    write_kept_curve(results, table)
    assert chart.stat().st_size > 1000
    assert "m2160_mom" in table.read_text(encoding="utf-8")


def test_roles_keep_four_seniors_on_the_short_window_and_two_days_on_the_minis():
    spec = make_split("none", "long-d2")
    one_long = [_state("long", 12_420, -1, held_days=2)]
    assert entry_veto("thirty", 30, one_long, one_long, spec, 0) == 1
    assert entry_veto("fortyfive", 45, [_state("long", 12_420, -1, held_days=1)], [_state("long", 12_420, -1, held_days=1)], spec, 0) == 0
    assert entry_veto("short", 525, one_long, one_long, spec, 0) == 0
    four = [
        _state("m1760", 1760, -1, held_days=1),
        _state("m2160", 2160, -1, held_days=1),
        _state("long", 12_420, -1, held_days=1),
        _state("momentum", 0, -1, held_days=1),
    ]
    windows = [state for state in four if state.name != "momentum"]
    assert entry_veto("short", 525, four, windows, spec, 0) == 1
    assert entry_veto("m1760", 1760, four, windows, spec, 0) == 0
    assert entry_veto("long", 12_420, four, windows, spec, 0) == 0
    assert entry_veto("momentum", 0, four, windows, spec, 0) == 0
    assert entry_veto("momentum", 0, one_long, one_long, spec, 0) == 1
    assert entry_veto("momentum", 0, [_state("thirty", 30, -1, held_days=3)], [_state("thirty", 30, -1, held_days=3)], spec, 0) == 0
    index = pd.date_range("2024-01-08 10:00", periods=40, freq="min", tz="Europe/Moscow")
    frame = pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.05,
            "volume": 100.0,
        },
        index=index,
    )
    result = run_eight({"CRH4": frame}, SPECS[0], {"CRH4": date(2024, 1, 8)})
    assert [stats.name for stats in result.strategies] == [
        "short",
        "long",
        "thirty",
        "fortyfive",
        "sixty",
        "m1760",
        "m2160",
        "momentum",
    ]
    assert sum(stats.cash for stats in result.strategies) == pytest.approx(START_SUM)
    assert result.curve
    assert result.dates[0] == date(2024, 1, 8)


def test_fortyfive_rule_stays_on_fortyfive_and_leaves_1760_free():
    spec = fortyfive_spec(
        "d0",
        "старшее, плюс сразу",
        RoleRule("quorum", quorum=1, hold_days=0, momentum_votes=False),
    )
    fresh = [_state("long", 12_420, -1, held_days=0)]
    ripe = [_state("long", 12_420, -1, held_days=2)]
    only_sixty = [_state("sixty", 60, -1, held_days=0)]
    assert spec.role("m1760").mode == "none"
    assert spec.role("momentum").mode == "none"
    assert spec.role("thirty").mode == "legacy"
    assert spec.role("sixty").hold_days == 0
    assert entry_veto("fortyfive", 45, fresh, fresh, spec, 0) == 1
    assert entry_veto("thirty", 30, fresh, fresh, spec, 0) == 0
    assert entry_veto("thirty", 30, ripe, ripe, spec, 0) == 1
    assert entry_veto("sixty", 60, fresh, fresh, spec, 0) == 1
    assert entry_veto("m1760", 1760, ripe, ripe, spec, 0) == 0
    assert entry_veto("momentum", 0, ripe, ripe, spec, 0) == 0
    leg = fortyfive_spec("leg", "старшее, 2 дня", RoleRule("legacy"))
    assert entry_veto("fortyfive", 45, fresh, fresh, leg, 0) == 0
    assert entry_veto("fortyfive", 45, ripe, ripe, leg, 0) == 1
    nearest = fortyfive_spec(
        "sixty-d0",
        "только 60, плюс сразу",
        RoleRule("quorum", quorum=1, hold_days=0, slower_only=False, momentum_votes=False, voters=frozenset({"sixty"})),
    )
    assert entry_veto("fortyfive", 45, fresh, fresh, nearest, 0) == 0
    assert entry_veto("fortyfive", 45, only_sixty, only_sixty, nearest, 0) == 1


def test_fortyfive_choice_refuses_a_gain_that_taxes_thirty():
    from trend import EightResult

    names = ("short", "long", "thirty", "fortyfive", "sixty", "m1760", "m2160", "momentum")

    def book(spec: object, nets: dict[str, float]) -> EightResult:
        strategies = [
            StrategyStats(name, [{"pnlcomm": nets[name], "secid": "CRM2"}], 0, 0, 0, 100_000 + nets[name], -1, 100_000)
            for name in names
        ]
        return EightResult(spec, strategies, -10, [], [])  # type: ignore[arg-type]

    flat = {name: 1_000.0 for name in names}
    taxed = dict(flat)
    taxed["fortyfive"] = 2_000.0
    taxed["thirty"] = 500.0
    safe = dict(flat)
    safe["fortyfive"] = 1_400.0
    shorter = dict(flat)
    shorter["fortyfive"] = 3_000.0
    base = next(spec for spec in SPECS if spec.id == "base")
    anchor = fortyfive_spec("leg", "старшее, 2 дня", RoleRule("legacy"))
    rich = fortyfive_spec("d0", "плюс сразу", RoleRule("quorum", quorum=1, hold_days=0, momentum_votes=False))
    costly = fortyfive_spec("none", "само", RoleRule("none"))
    mini = fortyfive_spec(
        "thirty-d2",
        "только 30",
        RoleRule("quorum", quorum=1, hold_days=2, slower_only=False, voters=frozenset({"thirty"})),
    )
    results = [
        book(base, flat),
        book(anchor, flat),
        book(rich, safe),
        book(costly, taxed),
        book(mini, shorter),
    ]
    assert choose_fortyfive(results).spec.id == "f45-d0"


def test_kept_fortyfive_yields_only_to_1760_after_two_days():
    spec = kept_split()
    wide = [_state("m1760", 1760, -1, held_days=2)]
    fresh = [_state("m1760", 1760, -1, held_days=1)]
    other = [_state("sixty", 60, -1, held_days=2)]
    assert spec.role("fortyfive").voters == frozenset({"m1760"})
    assert spec.role("thirty").mode == "legacy"
    assert entry_veto("fortyfive", 45, wide, wide, spec, 0) == 1
    assert entry_veto("fortyfive", 45, fresh, fresh, spec, 0) == 0
    assert entry_veto("fortyfive", 45, other, other, spec, 0) == 0
    assert entry_veto("m1760", 1760, wide, wide, spec, 0) == 0


def test_thirty_choice_refuses_a_gain_that_taxes_fortyfive():
    from trend import EightResult

    names = ("short", "long", "thirty", "fortyfive", "sixty", "m1760", "m2160", "momentum")

    def book(spec: object, nets: dict[str, float]) -> EightResult:
        strategies = [
            StrategyStats(name, [{"pnlcomm": nets[name], "secid": "CRM2"}], 0, 0, 0, 100_000 + nets[name], -1, 100_000)
            for name in names
        ]
        return EightResult(spec, strategies, -10, [], [])  # type: ignore[arg-type]

    flat = {name: 1_000.0 for name in names}
    taxed = dict(flat)
    taxed["thirty"] = 2_000.0
    taxed["fortyfive"] = 400.0
    safe = dict(flat)
    safe["thirty"] = 1_500.0
    base = next(spec for spec in SPECS if spec.id == "base")
    anchor = thirty_spec("leg", "старшее, 2 дня", RoleRule("legacy"))
    rich = thirty_spec("d0", "плюс сразу", RoleRule("quorum", quorum=1, hold_days=0, momentum_votes=False))
    costly = thirty_spec("none", "само", RoleRule("none"))
    changed = thirty_spec("m1760-d2", "только 1760", RoleRule("quorum", quorum=1, hold_days=2, voters=frozenset({"m1760"})))
    assert changed.role("fortyfive").voters == frozenset({"m1760"})
    assert entry_veto("thirty", 30, [_state("m1760", 1760, -1, held_days=2)], [_state("m1760", 1760, -1, held_days=2)], changed, 0) == 1
    assert entry_veto("thirty", 30, [_state("long", 12_420, -1, held_days=2)], [_state("long", 12_420, -1, held_days=2)], changed, 0) == 0
    results = [book(base, flat), book(anchor, flat), book(rich, safe), book(costly, taxed)]
    assert choose_thirty(results).spec.id == "t30-d0"


def test_momentum_rule_leaves_1760_free_and_blocks_only_momentum():
    spec = momentum_spec(
        "long-d2",
        "только длинное, 2 дня",
        RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False, voters=frozenset({"long"})),
    )
    one_long = [_state("long", 12_420, -1, held_days=2)]
    fresh_long = [_state("long", 12_420, -1, held_days=0)]
    wide = [_state("m2160", 2160, -1, held_days=2)]
    ripe_momentum = [_state("momentum", 0, -1, held_days=2)]
    assert spec.role("m1760").mode == "none"
    assert spec.role("momentum").voters == frozenset({"long"})
    assert entry_veto("momentum", 0, one_long, one_long, spec, 0) == 1
    assert entry_veto("momentum", 0, fresh_long, fresh_long, spec, 0) == 0
    assert entry_veto("momentum", 0, wide, wide, spec, 0) == 0
    assert entry_veto("m1760", 1760, one_long, one_long, spec, 0) == 0
    assert entry_veto("m2160", 2160, ripe_momentum, [], spec, 0) == 1
    assert entry_veto("sixty", 60, fresh_long, fresh_long, spec, 0) == 1
    assert entry_veto("thirty", 30, fresh_long, fresh_long, spec, 0) == 0
    alone = momentum_spec("none", "сам", RoleRule("none"))
    assert entry_veto("momentum", 0, one_long, one_long, alone, 0) == 0


def test_momentum_choice_refuses_a_gain_that_taxes_2160():
    from trend import EightResult

    names = ("short", "long", "thirty", "fortyfive", "sixty", "m1760", "m2160", "momentum")

    def book(spec: object, nets: dict[str, float]) -> EightResult:
        strategies = [
            StrategyStats(name, [{"pnlcomm": nets[name], "secid": "CRM2"}], 0, 0, 0, 100_000 + nets[name], -1, 100_000)
            for name in names
        ]
        return EightResult(spec, strategies, -10, [], [])  # type: ignore[arg-type]

    flat = {name: 1_000.0 for name in names}
    taxed = dict(flat)
    taxed["momentum"] = 2_000.0
    taxed["m2160"] = 500.0
    safe = dict(flat)
    safe["momentum"] = 1_400.0
    base = next(spec for spec in SPECS if spec.id == "base")
    anchor = momentum_spec("none", "сам", RoleRule("none"))
    rich = momentum_spec("long-d2", "только длинное", RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False))
    costly = momentum_spec("leg", "любое окно", RoleRule("legacy"))
    results = [book(base, flat), book(anchor, flat), book(rich, safe), book(costly, taxed)]
    assert choose_momentum(results).spec.id == "mom-long-d2"
