from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _num(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(float)
    return pd.Series(float(default), index=df.index, dtype="float64")


def _str(df: pd.DataFrame, names: list[str], default: str = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna(default).astype(str)
    return pd.Series(str(default), index=df.index, dtype="object")


def _time_hhmm(df: pd.DataFrame) -> pd.Series:
    if "timestamp_ny" in df.columns:
        raw = df["timestamp_ny"].astype(str)
        # accepts either full ISO string or HH:MM-like string
        out = raw.str.extract(r"(\d{2}:\d{2})", expand=False)
        if out.notna().any():
            return out.fillna("")
    for name in ["timestamp", "entry_time", "signal_time"]:
        if name in df.columns:
            ts = pd.to_datetime(df[name], utc=True, errors="coerce")
            try:
                return ts.dt.tz_convert("America/New_York").dt.strftime("%H:%M").fillna("")
            except Exception:
                return ts.dt.strftime("%H:%M").fillna("")
    return pd.Series("", index=df.index)


def _contains(series: pd.Series, text: str) -> pd.Series:
    return series.astype(str).str.lower().str.contains(text.lower(), regex=False, na=False)


def apply_decision_time_pattern_scorer(df: pd.DataFrame, params: Any) -> pd.DataFrame:
    """V37.9 live-safe decision-time indicator pattern scorer.

    This filter is deliberately based only on fields known at the signal candle:
    side, trigger/setup, candle classification, RVOL, relative strength, VWAP
    extension, daily ATR, gap, QQQ move, and candidate score. It does not use
    future P&L, MFE/MAE, exits, or any later bars to decide whether a candidate
    is valid.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    mode = str(getattr(params, "v379_pattern_mode", "balanced_vwap_prevhigh") or "balanced_vwap_prevhigh").lower()

    side = _str(out, ["side"], "").str.lower()
    short = side.eq("short")
    long = side.eq("long")
    trigger = _str(out, ["trigger_type", "event", "setup_family"], "").str.lower()
    candle = _str(out, ["entry_candle_pattern", "candle_pattern_primary", "pat"], "").str.lower()
    hhmm = _time_hhmm(out)

    rvol = _num(out, ["rvol_time_of_day", "rvol_tod"], 0.0)
    day_rs = _num(out, ["day_relative_strength", "rs_open"], 0.0)
    open_rs = _num(out, ["open_relative_strength", "rs_open"], 0.0)
    vwap = _num(out, ["vwap_extension_atr", "vwap_ext_atr"], 0.0)
    daily_atr = _num(out, ["daily_atr14_percent", "daily_atr_pct"], 0.0)
    gap = _num(out, ["gap_percent", "gap_pct"], 0.0)
    qqq = _num(out, ["qqq_day_change_percent", "qqq_change_from_open", "qqq_chg_open"], 0.0)
    orig_score = _num(out, ["candidate_score", "score", "robust_score"], 0.0)

    dir_rs = pd.Series(np.where(short, -day_rs, day_rs), index=out.index, dtype="float64")
    dir_open_rs = pd.Series(np.where(short, -open_rs, open_rs), index=out.index, dtype="float64")
    dir_vwap = pd.Series(np.where(short, -vwap, vwap), index=out.index, dtype="float64")
    abs_vwap = vwap.abs()
    abs_gap = gap.abs()
    abs_qqq = qqq.abs()
    abs_rs = day_rs.abs()

    side_continuation = (
        candle.str.contains("side_continuation", na=False)
        | (short & candle.str.contains("bearish_continuation|bearish_inside_breakout", regex=True, na=False))
        | (long & candle.str.contains("bullish_continuation|bullish_inside_breakout", regex=True, na=False))
    )
    side_rejection = (
        candle.str.contains("side_rejection", na=False)
        | (short & candle.str.contains("bearish_rejection|bearish_engulfing", regex=True, na=False))
        | (long & candle.str.contains("bullish_rejection|bullish_engulfing", regex=True, na=False))
    )
    side_engulfing = (
        candle.str.contains("side_engulfing", na=False)
        | (short & candle.str.contains("bearish_engulfing", na=False))
        | (long & candle.str.contains("bullish_engulfing", na=False))
    )

    score = pd.Series(0.0, index=out.index)
    candle_component = pd.Series(0.0, index=out.index)
    candle_component += np.where(side_continuation, 3.0, 0.0)
    candle_component += np.where(side_rejection, 1.0, 0.0)
    candle_component += np.where(side_engulfing, -1.0, 0.0)
    rvol_component = pd.Series(0.0, index=out.index)
    rvol_component += np.where((rvol >= 0.85) & (rvol <= 1.30), 2.0, 0.0)
    rvol_component += np.where(rvol > 1.60, 1.0, 0.0)
    rvol_component += np.where(rvol < 0.85, -2.0, 0.0)
    rs_component = pd.Series(0.0, index=out.index)
    rs_component += np.where(dir_rs > 0.50, 2.0, 0.0)
    rs_component += np.where(day_rs.between(-0.40, 0.40), -2.0, 0.0)
    vwap_component = pd.Series(0.0, index=out.index)
    vwap_component += np.where(short & (vwap < -1.00), 2.0, 0.0)
    vwap_component += np.where(short & (vwap >= -1.00) & (vwap < -0.50), 1.0, 0.0)
    vwap_component += np.where(long & vwap.between(-0.80, 1.50), 1.0, 0.0)
    vwap_component += np.where(long & (vwap > 2.00), -1.0, 0.0)
    old_score_component = pd.Series(np.where(orig_score.between(0, 15), 1.0, 0.0), index=out.index)
    score = candle_component + rvol_component + rs_component + vwap_component + old_score_component

    # Common trigger sets.
    is_vwap = _contains(trigger, "vwap_pullback")
    is_prev_high = _contains(trigger, "prev_high_sweep")
    is_orb = _contains(trigger, "10_orb")
    is_late = _contains(trigger, "late_trend")
    is_gap = _contains(trigger, "gap_cont") | _contains(trigger, "gap_continuation") | _contains(trigger, "gap")
    is_prev_low = _contains(trigger, "prev_low_sweep")

    if mode in {"baseline_user", "user_pattern", "v379_baseline"}:
        passed = (
            (side_continuation | side_rejection)
            & (rvol >= 0.85)
            & (~day_rs.between(-0.40, 0.40))
            & (
                (long & (day_rs > 0.50) & vwap.between(-0.80, 1.50))
                | (short & (day_rs < -0.50) & (vwap < -0.50))
            )
        )
        mode_name = "v379_baseline_user_pattern"
    elif mode in {"short_vwap_morning", "short_vwap", "v379_short_vwap"}:
        passed = (
            short & is_vwap
            & (hhmm >= "10:30") & (hhmm <= "12:00")
            & rvol.between(0.85, 2.00)
            & (abs_rs >= 0.50)
            & dir_vwap.between(0.25, 1.50)
            & (abs_vwap <= 2.00)
            & daily_atr.between(5.00, 6.00)
            & orig_score.between(5.00, 40.00)
            & (abs_qqq <= 3.00)
        )
        mode_name = "v379_short_vwap_morning"
    elif mode in {"orb_vwap", "v379_orb_vwap"}:
        passed = (
            (is_vwap | is_orb)
            & (hhmm >= "10:00") & (hhmm <= "13:00")
            & rvol.between(0.85, 1.30)
            & (dir_rs >= 1.00)
            & (abs_rs >= 0.50)
            & dir_vwap.between(-0.50, 1.00)
            & (orig_score >= 15.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 5.00)
        )
        mode_name = "v379_orb_vwap"

    elif mode in {"v38_balanced", "v38_2_balanced", "balanced_active", "v382_balanced"}:
        # V38.2 balanced active mode.  This is a less restrictive pattern gate
        # designed to find more opportunities than V38 stable while still using
        # only decision-time fields.  It comes from the best multi-period raw-live
        # search pocket: 10:00-11:00 ET, VWAP/ORB/late-trend events, non-engulfing
        # candles, moderate+ RVOL, non-extreme directional RS and VWAP context.
        not_engulf = ~side_engulfing
        passed = (
            (is_vwap | is_orb | is_late)
            & not_engulf
            & (hhmm >= "10:00") & (hhmm <= "11:00")
            & (rvol >= 1.00)
            & dir_rs.between(-2.00, 2.00)
            & dir_vwap.between(-2.00, 3.00)
            & daily_atr.between(0.00, 8.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 5.00)
        )
        # Scoring favors candle quality and controlled directional movement, but
        # does not hard-block low candidate_score values.
        score = (
            candle_component
            + rvol_component
            + rs_component
            + vwap_component
            + np.where(abs_rs.between(0.50, 2.50), 1.0, 0.0)
            + np.where(dir_vwap.between(0.00, 2.00), 1.0, 0.0)
            + np.where(abs_vwap <= 2.50, 0.5, 0.0)
        )
        mode_name = "v38_2_balanced_active"
    elif mode in {"v38_medium", "v38_2_medium", "more_trades", "v382_more_trades"}:
        # V38.2 medium-frequency research mode.  Broader than balanced active:
        # 09:45-11:00, all core setups except previous-low reclaim, continuation
        # candles only, directional VWAP confirmation.  This intentionally finds
        # more trades, but should be treated as experimental and validated before
        # paper/live use.
        core_no_prevlow = (is_vwap | is_orb | is_gap | is_late | is_prev_high)
        passed = (
            core_no_prevlow
            & side_continuation
            & (hhmm >= "09:45") & (hhmm <= "11:00")
            & (rvol >= 1.00)
            & dir_rs.between(-2.00, 3.00)
            & dir_vwap.between(0.00, 1.50)
            & daily_atr.between(1.00, 8.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 5.00)
        )
        score = (
            candle_component
            + rvol_component
            + rs_component
            + vwap_component
            + np.where(dir_vwap.between(0.25, 1.25), 1.5, 0.0)
            + np.where(abs_rs >= 0.30, 1.0, 0.0)
        )
        mode_name = "v38_2_more_trades"
    elif mode in {"v38_active", "active", "active_vwap_orb", "v38_active_vwap_orb"}:
        # V38 active mode: accepts more opportunities than V37.9 while keeping
        # the same causal decision-time field set. This was chosen as a
        # higher-frequency research mode, not as a final robust live edge.
        passed = (
            (is_vwap | is_orb | is_late | is_gap)
            & (side_continuation | side_rejection)
            & (hhmm >= "09:35") & (hhmm <= "11:30")
            & rvol.between(0.70, 3.50)
            & (abs_rs >= 0.35)
            & dir_rs.between(-1.50, 3.50)
            & dir_vwap.between(-1.00, 2.00)
            & daily_atr.between(2.00, 10.00)
            & orig_score.between(0.00, 999.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 5.00)
        )
        mode_name = "v38_active_vwap_orb"
    elif mode in {"v38_stable", "stable", "stable_no_prevlow", "v38_stable_no_prevlow"}:
        # V38 stable mode: fewer trades than active mode, but it was positive in
        # more calendar-year/partial-year slices in the raw-live candidate test.
        passed = (
            (is_vwap | is_prev_high | is_orb | is_late | is_gap)
            & side_continuation
            & (hhmm >= "09:35") & (hhmm <= "11:00")
            & rvol.between(1.20, 3.00)
            & (abs_rs >= 0.75)
            & (~day_rs.between(-0.40, 0.40))
            & dir_rs.between(0.00, 1.50)
            & dir_vwap.between(0.00, 3.00)
            & daily_atr.between(2.00, 8.00)
            & orig_score.between(5.00, 999.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 5.00)
        )
        mode_name = "v38_stable_no_prevlow"

    elif mode in {"v385_adaptive_plus", "v385_scaled_adaptive", "v38_5_adaptive_plus", "v38_5_scaled", "adaptive_plus_v385"}:
        # V38.5 adaptive-plus router.
        #
        # This is based on the user's longer V38.3 raw-replay report.  The report
        # showed that V38.3 was the best base, but that damage was concentrated in
        # a few repeated contexts: MA long VWAP pullbacks, INTC short late-trend,
        # COIN shorts, and entries at/after 12:00 ET.  The profitable, scalable
        # pockets were AMD/MU short VWAP weakness, CRM/XOM long ORB, selected
        # COIN long VWAP, and a single XOM long late-trend exception.  All rules
        # below use only decision-time features.
        sym = _str(out, ["symbol"], "").str.upper()

        hard_damage_ma_long_vwap = sym.eq("MA") & long & is_vwap
        hard_damage_intc_short_late = sym.eq("INTC") & short & is_late
        hard_damage_coin_short = sym.eq("COIN") & short
        hard_damage_after_noon = hhmm >= "12:00"
        hard_damage = (
            hard_damage_ma_long_vwap
            | hard_damage_intc_short_late
            | hard_damage_coin_short
            | hard_damage_after_noon
        )

        # Main scalable short pocket: AMD/MU short VWAP weakness.  This keeps the
        # V38.3 winners while limiting panic/late/huge-gap shorts.
        amd_mu_short_vwap = (
            sym.isin(["AMD", "MU"])
            & short & is_vwap
            & (hhmm >= "09:35") & (hhmm <= "11:55")
            & (rvol >= 0.70)
            & daily_atr.between(3.00, 8.75)
            & dir_rs.between(0.45, 4.25)
            & dir_vwap.between(0.05, 1.80)
            & (abs_gap <= 2.75)
            & (abs_qqq <= 2.25)
            & (~side_engulfing)
        )

        # Strong long pocket: CRM/XOM 10:00 opening-range break/continuation.
        crm_xom_long_orb = (
            sym.isin(["CRM", "XOM"])
            & long & is_orb
            & (hhmm >= "09:45") & (hhmm <= "10:55")
            & (rvol >= 0.85)
            & (
                (sym.eq("CRM") & (daily_atr >= 3.00))
                | (sym.eq("XOM") & (daily_atr >= 2.00))
            )
            & (daily_atr <= 5.50)
            & (dir_rs >= 0.30)
            & dir_vwap.between(0.45, 2.05)
            & (abs_gap <= 2.25)
            & (abs_qqq <= 2.25)
            & (~side_engulfing)
        )

        # COIN long VWAP worked when it was not engulfing/chasing and before 11:05.
        coin_long_vwap = (
            sym.eq("COIN")
            & long & is_vwap
            & (hhmm >= "10:00") & (hhmm <= "11:00")
            & rvol.between(1.20, 3.25)
            & (dir_rs >= 0.50)
            & dir_vwap.between(0.00, 1.05)
            & (abs_gap <= 2.00)
            & (abs_qqq <= 1.50)
            & (~side_engulfing)
        )

        # Very small but clean XOM late trend exception.  It is kept separate so
        # the model does not reopen generic late-trend longs.
        xom_late_long_exception = (
            sym.eq("XOM")
            & long & is_late
            & (hhmm >= "11:00") & (hhmm <= "11:30")
            & (rvol >= 1.20)
            & daily_atr.between(1.50, 2.25)
            & dir_rs.between(0.80, 2.25)
            & dir_vwap.between(0.80, 1.75)
            & (qqq <= 0.00)
            & (abs_gap <= 1.50)
        )

        passed = (
            amd_mu_short_vwap
            | crm_xom_long_orb
            | coin_long_vwap
            | xom_late_long_exception
        ) & (~hard_damage)

        base_quality = (
            np.where(side_continuation, 30.0, 0.0)
            + np.where(side_rejection, 8.0, 0.0)
            - np.where(side_engulfing, 25.0, 0.0)
            + np.where(rvol.between(0.85, 1.30), 20.0, 0.0)
            + np.where(rvol > 1.60, 10.0, 0.0)
            + np.where(dir_rs.between(0.50, 2.25), 20.0, 0.0)
            + np.where(dir_vwap.between(0.25, 1.60), 18.0, 0.0)
            + np.where(abs_qqq <= 1.50, 5.0, 0.0)
            + orig_score * 0.05
        )
        rule_bonus = (
            np.where(amd_mu_short_vwap, 70.0, 0.0)
            + np.where(crm_xom_long_orb, 85.0, 0.0)
            + np.where(coin_long_vwap, 60.0, 0.0)
            + np.where(xom_late_long_exception, 50.0, 0.0)
        )
        damage_penalty = (
            np.where(hard_damage_ma_long_vwap, 100.0, 0.0)
            + np.where(hard_damage_intc_short_late, 100.0, 0.0)
            + np.where(hard_damage_coin_short, 100.0, 0.0)
            + np.where(hard_damage_after_noon, 80.0, 0.0)
        )
        score = pd.Series(base_quality + rule_bonus - damage_penalty, index=out.index, dtype="float64")
        mode_name = "v385_adaptive_plus_damage_controlled"
        out["v385_rule_amd_mu_short_vwap"] = amd_mu_short_vwap.astype(bool)
        out["v385_rule_crm_xom_long_orb"] = crm_xom_long_orb.astype(bool)
        out["v385_rule_coin_long_vwap"] = coin_long_vwap.astype(bool)
        out["v385_rule_xom_late_long_exception"] = xom_late_long_exception.astype(bool)
        out["v385_hard_damage_block"] = hard_damage.astype(bool)
        out["v385_block_ma_long_vwap"] = hard_damage_ma_long_vwap.astype(bool)
        out["v385_block_intc_short_late"] = hard_damage_intc_short_late.astype(bool)
        out["v385_block_coin_short"] = hard_damage_coin_short.astype(bool)
        out["v385_block_after_noon"] = hard_damage_after_noon.astype(bool)
        out["v385_quality_score"] = score
        out["v385_rule_name"] = np.select(
            [amd_mu_short_vwap, crm_xom_long_orb, coin_long_vwap, xom_late_long_exception, hard_damage_ma_long_vwap, hard_damage_intc_short_late, hard_damage_coin_short, hard_damage_after_noon],
            ["amd_mu_short_vwap", "crm_xom_long_orb", "coin_long_vwap", "xom_late_long_exception", "blocked_ma_long_vwap", "blocked_intc_short_late", "blocked_coin_short", "blocked_after_noon"],
            default="none",
        )

    elif mode in {"v384_failure_reversal", "v384_failure_aware", "failure_aware_reversal", "v38_4_failure_aware"}:
        # V38.4 failure-aware router.
        # This is not a hindsight reversal engine. It uses the failed high-activity
        # reports as negative examples and keeps only candidates whose current
        # decision-time state looks like the historically better side/setup zones.
        # If a row is on the wrong side of its own indicators, it is blocked and
        # marked as an opposite-watch candidate; it is not flipped unless the normal
        # signal generator has produced a candidate in the opposite direction.
        is_short_vwap = short & is_vwap
        is_long_vwap = long & is_vwap
        is_long_orb = long & is_orb
        is_long_late = long & is_late
        is_short_late = short & is_late

        # Opposite-side warnings learned from the broad failed reports.
        long_looks_short = long & (
            (day_rs < -0.40) | ((qqq < -0.25) & (open_rs < 0.25)) | (vwap < -0.50)
        )
        short_looks_long = short & (
            (day_rs > 0.40) | ((qqq > 0.25) & (open_rs > -0.25)) | (vwap > 0.50)
        )

        # Broad failure archetypes from uploaded reports:
        # - AMD/MU generic short VWAP was high-frequency but negative unless very controlled.
        # - MA long VWAP and XOM long ORB/late were persistent drags.
        # - inside-breakout/engulfing variants were unstable outside special contexts.
        sym = _str(out, ["symbol"], "").str.upper()
        failed_amd_short_vwap = is_short_vwap & sym.eq("AMD")
        failed_ma_long_vwap = is_long_vwap & sym.eq("MA")
        failed_xom_long_momentum = long & sym.eq("XOM") & (is_orb | is_late)
        failed_inside_breakout = candle.str.contains("inside_breakout", na=False) & ~(sym.isin(["CRM", "MU"]))
        failed_engulfing_generic = side_engulfing & ~(sym.isin(["CRM", "MU"]))

        negative_archetype = (
            failed_amd_short_vwap | failed_ma_long_vwap | failed_xom_long_momentum
            | failed_inside_breakout | failed_engulfing_generic
            | long_looks_short | short_looks_long
        )

        # Positive pockets preserved from the good reports and the previous stable modes.
        # MU short vwap was the cleanest short pocket; CRM long ORB/VWAP was the cleanest long pocket.
        mu_short_vwap_quality = (
            short & sym.eq("MU") & is_vwap
            & (hhmm >= "09:35") & (hhmm <= "12:00")
            & rvol.between(0.85, 2.75)
            & (dir_rs >= 0.50)
            & dir_vwap.between(0.50, 2.25)
            & daily_atr.between(2.00, 10.00)
            & (abs_qqq <= 3.00)
            & (~failed_inside_breakout)
        )
        crm_long_quality = (
            long & sym.eq("CRM") & (is_orb | is_vwap)
            & (hhmm >= "09:35") & (hhmm <= "12:00")
            & (rvol >= 0.85)
            & (dir_rs >= 0.30)
            & dir_vwap.between(-0.50, 2.50)
            & daily_atr.between(1.00, 8.00)
            & (abs_qqq <= 3.00)
        )
        intc_short_late_quality = (
            short & sym.eq("INTC") & is_late
            & (hhmm >= "10:00") & (hhmm <= "14:00")
            & rvol.between(0.70, 3.00)
            & (dir_rs >= 0.35)
            & dir_vwap.between(0.75, 3.00)
            & daily_atr.between(2.00, 7.00)
            & (abs_qqq <= 3.00)
        )

        # General quality rules that avoid the broad-failure zones.
        # These are deliberately side-aware and use directional-normalized fields.
        general_short_quality = (
            short & (is_vwap | is_prev_high | is_late)
            & (hhmm >= "09:35") & (hhmm <= "12:00")
            & rvol.between(0.85, 2.75)
            & (dir_rs >= 0.60)
            & (dir_open_rs >= -0.25)
            & dir_vwap.between(0.50, 2.50)
            & daily_atr.between(2.00, 10.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 5.00)
            & side_continuation
            & (~sym.isin(["AMD", "QCOM"]))
        )
        general_long_quality = (
            long & (is_orb | is_vwap | is_late)
            & (hhmm >= "09:35") & (hhmm <= "12:00")
            & rvol.between(0.85, 2.50)
            & (dir_rs >= 0.60)
            & (dir_open_rs >= -0.25)
            & dir_vwap.between(-0.50, 2.00)
            & daily_atr.between(1.00, 8.00)
            & (abs_qqq <= 2.50)
            & (abs_gap <= 5.00)
            & (side_continuation | side_rejection)
            & (~sym.isin(["MA", "XOM"]))
        )

        # If the candidate has opposite-side evidence, do not take it on this side.
        # A separate candidate in the opposite direction can still pass its own rules.
        passed = (
            mu_short_vwap_quality | crm_long_quality | intc_short_late_quality
            | general_short_quality | general_long_quality
        ) & (~negative_archetype | mu_short_vwap_quality | crm_long_quality | intc_short_late_quality)

        failure_score = pd.Series(0.0, index=out.index)
        failure_score += np.where(long_looks_short | short_looks_long, 50.0, 0.0)
        failure_score += np.where(failed_amd_short_vwap | failed_ma_long_vwap | failed_xom_long_momentum, 35.0, 0.0)
        failure_score += np.where(failed_inside_breakout | failed_engulfing_generic, 20.0, 0.0)

        quality_score = (
            np.where(side_continuation, 30.0, 0.0)
            + np.where(side_rejection, 10.0, 0.0)
            + np.where(rvol.between(0.85, 1.30), 20.0, 0.0)
            + np.where(rvol > 1.60, 10.0, 0.0)
            + np.where(dir_rs >= 0.60, 20.0, 0.0)
            + np.where(dir_open_rs >= 0.00, 10.0, 0.0)
            + np.where(dir_vwap.between(0.50, 2.00), 20.0, 0.0)
            + np.where(abs_qqq <= 1.50, 5.0, 0.0)
            + orig_score * 0.05
            - failure_score
        )
        score = pd.Series(quality_score, index=out.index, dtype="float64")
        mode_name = "v384_failure_aware_reversal_router"
        out["v384_negative_archetype"] = negative_archetype.astype(bool)
        out["v384_long_looks_short"] = long_looks_short.astype(bool)
        out["v384_short_looks_long"] = short_looks_long.astype(bool)
        out["v384_failure_score"] = failure_score
        out["v384_quality_score"] = score
        out["v384_rule_name"] = np.select(
            [mu_short_vwap_quality, crm_long_quality, intc_short_late_quality, general_short_quality, general_long_quality, long_looks_short, short_looks_long],
            ["mu_short_vwap_quality", "crm_long_or_vwap_quality", "intc_short_late_quality", "general_short_quality", "general_long_quality", "reject_long_opposite_short_watch", "reject_short_opposite_long_watch"],
            default="none",
        )

    elif mode in {"v383_adaptive", "v383_regime_adaptive", "adaptive_regime", "v38_3_adaptive", "v383_adaptive_composite", "v38_3_regime_adaptive", "regime_adaptive_v383"}:
        # V38.3 adaptive composite: learned from comparing the user's uploaded
        # live/raw strategy reports and an out-of-sample candidate simulation.
        #
        # The broad active filters produced many trades but lost money.  The stable
        # filters were profitable but too sparse.  This composite keeps three
        # decision-time patterns that were positive across the checked 2022-2026
        # candidate replay years: core high-RVOL VWAP/ORB/late momentum, a long
        # VWAP/ORB strength pocket, and a stricter short late-morning weakness
        # pocket.  It uses only signal-time fields.
        rule_core = (
            (is_vwap | is_orb | is_late)
            & (hhmm >= "09:35") & (hhmm <= "12:00")
            & (rvol >= 1.50)
            & (daily_atr >= 1.00)
            & (abs_rs >= 0.75)
            & dir_rs.between(0.00, 2.00)
            & dir_vwap.between(0.75, 3.00)
            & (abs_vwap <= 1.50)
            & (abs_qqq <= 4.00)
            & (orig_score >= 5.00)
        )
        rule_long_vwap_orb = (
            long
            & (is_vwap | is_orb)
            & (hhmm >= "09:40") & (hhmm <= "11:00")
            & (rvol >= 1.00)
            & (daily_atr >= 2.00)
            & (abs_rs >= 0.30)
            & dir_rs.between(-1.00, 3.00)
            & dir_vwap.between(-1.00, 2.00)
            & (abs_vwap <= 3.00)
            & (abs_qqq <= 2.00)
            & (orig_score >= 20.00)
            & (~side_engulfing)
        )
        rule_short_late_weakness = (
            short
            & (is_vwap | is_orb | is_late)
            & (hhmm >= "10:30") & (hhmm <= "12:00")
            & rvol.between(0.70, 3.00)
            & daily_atr.between(5.00, 10.00)
            & (dir_rs >= 0.50)
            & dir_vwap.between(-1.00, 2.00)
            & (abs_vwap <= 1.50)
            & (abs_qqq <= 2.00)
            & (abs_gap <= 2.50)
            & (~side_engulfing)
        )
        passed = rule_core | rule_long_vwap_orb | rule_short_late_weakness

        pattern_rank = (
            np.where(side_continuation, 30.0, 0.0)
            + np.where(side_rejection, 10.0, 0.0)
            - np.where(side_engulfing, 20.0, 0.0)
            + np.where(rvol.between(0.85, 1.30), 20.0, 0.0)
            + np.where(rvol > 1.60, 10.0, 0.0)
            + np.where(abs_rs >= 0.50, 15.0, 0.0)
            + np.where(dir_vwap.between(0.50, 2.00), 15.0, 0.0)
            + orig_score * 0.10
        )
        dir_rank = orig_score + (10.0 * rvol) + (5.0 * dir_vwap) + (3.0 * dir_rs)
        adaptive_rank = pd.Series(-999999.0, index=out.index, dtype="float64")
        adaptive_rank = adaptive_rank.mask(rule_core, orig_score)
        adaptive_rank = pd.Series(np.maximum(adaptive_rank, np.where(rule_long_vwap_orb, pattern_rank, -999999.0)), index=out.index)
        adaptive_rank = pd.Series(np.maximum(adaptive_rank, np.where(rule_short_late_weakness, dir_rank, -999999.0)), index=out.index)
        score = adaptive_rank.where(passed, score)
        mode_name = "v383_adaptive_composite"
        out["v383_rule_core"] = rule_core.astype(bool)
        out["v383_rule_long_vwap_orb"] = rule_long_vwap_orb.astype(bool)
        out["v383_rule_short_late_weakness"] = rule_short_late_weakness.astype(bool)
        out["v383_rule_name"] = np.select(
            [rule_core, rule_long_vwap_orb, rule_short_late_weakness],
            ["core_high_rvol_vwap_rs", "long_vwap_orb", "short_late_weakness"],
            default="none",
        )
    elif mode in {"v382_active_plus", "active_plus", "v38_active_plus", "v382_balanced_active"}:
        # V38.2 active-plus mode.  More active than stable, but still stricter
        # than the loose active mode that lost money in the uploaded reports.
        passed = (
            (is_vwap | is_orb | is_late)
            & (~side_engulfing)
            & (hhmm >= "10:00") & (hhmm <= "11:00")
            & (rvol >= 1.00)
            & (abs_rs >= 1.00)
            & dir_rs.between(-2.00, 2.00)
            & dir_vwap.between(-2.00, 3.00)
            & daily_atr.between(0.00, 8.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 5.00)
        )
        mode_name = "v382_active_plus"
    elif mode in {"v382_more_trades", "more_trades", "v382_high_activity"}:
        # Research-only higher-activity mode.  It is intentionally broader and
        # should not be treated as the default live preset without validation.
        passed = (
            (is_vwap | is_orb | is_prev_high | is_late | is_gap)
            & (~is_prev_low)
            & side_continuation
            & (hhmm >= "09:45") & (hhmm <= "11:00")
            & (rvol >= 1.00)
            & (abs_rs >= 0.30)
            & dir_rs.between(-2.00, 3.00)
            & dir_vwap.between(0.00, 1.50)
            & daily_atr.between(1.00, 8.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 5.00)
        )
        mode_name = "v382_more_trades_research"
    else:
        # Default chosen from raw-live candidate simulation. It is positive in
        # train/validation/holdout but still should be treated as experimental.
        passed = (
            (is_vwap | is_prev_high)
            & (hhmm >= "09:35") & (hhmm <= "12:00")
            & (rvol >= 1.50)
            & dir_rs.between(0.50, 2.00)
            & (abs_rs >= 1.00)
            & dir_vwap.between(0.50, 2.00)
            & (abs_vwap <= 1.50)
            & daily_atr.between(1.00, 8.00)
            & orig_score.between(10.00, 40.00)
            & (abs_qqq <= 3.00)
            & (abs_gap <= 2.50)
        )
        mode_name = "v379_balanced_vwap_prevhigh"

    out["v379_pattern_mode"] = mode_name
    out["v379_pattern_match"] = passed.fillna(False).astype(bool)
    out["v379_original_score"] = orig_score
    out["v379_candle_component"] = candle_component
    out["v379_rvol_component"] = rvol_component
    out["v379_rs_component"] = rs_component
    out["v379_vwap_component"] = vwap_component
    out["v379_old_score_component"] = old_score_component
    out["v379_pattern_score"] = score
    out["v379_directional_rs"] = dir_rs
    out["v379_directional_open_rs"] = dir_open_rs
    out["v379_directional_vwap_atr"] = dir_vwap
    out["v379_abs_vwap_atr"] = abs_vwap
    out["v379_abs_gap_percent"] = abs_gap
    out["v379_abs_qqq_change"] = abs_qqq
    out["v379_side_continuation"] = side_continuation
    out["v379_side_rejection"] = side_rejection
    out["v379_side_engulfing"] = side_engulfing
    out["v379_reason"] = (
        "mode=" + mode_name
        + "|candle=" + candle.astype(str)
        + "|rvol=" + rvol.round(3).astype(str)
        + "|day_rs=" + day_rs.round(3).astype(str)
        + "|dir_rs=" + dir_rs.round(3).astype(str)
        + "|vwap=" + vwap.round(3).astype(str)
        + "|dir_vwap=" + dir_vwap.round(3).astype(str)
        + "|daily_atr=" + daily_atr.round(3).astype(str)
        + "|old_score=" + orig_score.round(3).astype(str)
        + "|pattern_score=" + score.round(3).astype(str)
    )
    if "v383_rule_name" in out.columns:
        out["v379_reason"] = out["v379_reason"] + "|adaptive_rule=" + out["v383_rule_name"].astype(str)
    if "v384_rule_name" in out.columns:
        out["v379_reason"] = out["v379_reason"] + "|failure_aware_rule=" + out["v384_rule_name"].astype(str) + "|failure_score=" + out.get("v384_failure_score", pd.Series(0, index=out.index)).round(3).astype(str)
    if "v385_rule_name" in out.columns:
        out["v379_reason"] = out["v379_reason"] + "|v385_rule=" + out["v385_rule_name"].astype(str) + "|v385_damage_block=" + out.get("v385_hard_damage_block", pd.Series(False, index=out.index)).astype(str) + "|v385_quality=" + out.get("v385_quality_score", pd.Series(0, index=out.index)).round(3).astype(str)
    # Rank by pattern score first, then old score. Preserve original score for audit.
    rank_score = (score * 100.0) + orig_score.fillna(0.0)
    out["v379_rank_score"] = rank_score
    out["score"] = rank_score
    out["candidate_score"] = rank_score

    # For raw signal frames, block alert booleans. For candidate frames, return only matches.
    if "buy_alert" in out.columns:
        out["buy_alert"] = out["buy_alert"].fillna(False).astype(bool) & out["v379_pattern_match"]
    if "short_alert" in out.columns:
        out["short_alert"] = out["short_alert"].fillna(False).astype(bool) & out["v379_pattern_match"]
    if "buy_alert" not in out.columns and "short_alert" not in out.columns:
        out = out.loc[out["v379_pattern_match"]].copy()
    return out
