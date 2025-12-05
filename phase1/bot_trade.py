import math
from collections import deque
from typing import Optional

# ============================================================================
# Capital-preserving trading bot
# - Risk-on only when all three are bullish:
#   (1) SMA(15) > SMA(50) by margin
#   (2) MACD(12,26,9) > signal
#   (3) Recent momentum positive
# - Otherwise stay near cash
# - Keep volatility targeting and gentle RSI nudges
# - Preserve make_decision signature/returns
# ============================================================================

# Persistent state
history: list[dict[str, float]] = []
_prices_short = deque(maxlen=15)  # SMA(15)
_prices_long = deque(maxlen=50)  # SMA(50)
_returns = deque(maxlen=30)  # recent returns for momentum/vol

# MACD state (EMA values)
_macd_ema_fast: Optional[float] = None
_macd_ema_slow: Optional[float] = None
_macd_signal: Optional[float] = None

_prev_alloc: float = 0.5

# Parameters
# Base parameters (will be adapted per regime)
BASE_MIN_ALLOC = 0.15
BASE_MAX_ALLOC = 0.85
TURNOVER_THRESHOLD = 0.02
BASE_HYSTERESIS_PCT = 0.001  # 0.10% of price
BASE_TARGET_VOL = 0.035
BASE_SCALE_FLOOR = 0.5
BASE_MOMENTUM_LOOKBACK = 10  # recent momentum window

# Regime adaptation ranges
# In high volatility: widen hysteresis, tighten bounds, increase scale floor, lengthen momentum lookback
HI_VOL_HYSTERESIS_PCT = 0.0015  # 0.15%
HI_VOL_MIN_ALLOC = 0.20
HI_VOL_MAX_ALLOC = 0.80
HI_VOL_SCALE_FLOOR = 0.6
HI_VOL_MOMENTUM_LOOKBACK = 20

# In low volatility: narrower hysteresis, allow wider bounds, lower scale floor, shorter momentum lookback
LO_VOL_HYSTERESIS_PCT = 0.0008  # 0.08%
LO_VOL_MIN_ALLOC = 0.15
LO_VOL_MAX_ALLOC = 0.85
LO_VOL_SCALE_FLOOR = 0.5
LO_VOL_MOMENTUM_LOOKBACK = 8

# Volatility regime thresholds (using ATR-like proxy)
VOL_LOW_THRESH = 0.003  # low vol if atr < 0.3%
VOL_HIGH_THRESH = 0.008  # high vol if atr > 0.8%

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _sma(values: deque) -> Optional[float]:
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def _update_state(price: float):
    if len(history) >= 1:
        prev = history[-1]["price"]
        r = price / prev - 1.0
        _returns.append(r)


def _ema(prev: Optional[float], value: float, alpha: float) -> float:
    return value if prev is None else (alpha * value + (1.0 - alpha) * prev)


def _update_macd(price: float):
    global _macd_ema_fast, _macd_ema_slow, _macd_signal
    alpha_fast = 2.0 / (12 + 1)
    alpha_slow = 2.0 / (26 + 1)
    _macd_ema_fast = _ema(_macd_ema_fast, price, alpha_fast)
    _macd_ema_slow = _ema(_macd_ema_slow, price, alpha_slow)
    macd = (
        (_macd_ema_fast - _macd_ema_slow)
        if (_macd_ema_fast is not None and _macd_ema_slow is not None)
        else 0.0
    )
    alpha_signal = 2.0 / (9 + 1)
    _macd_signal = _ema(_macd_signal, macd, alpha_signal)
    return macd, (_macd_signal if _macd_signal is not None else 0.0)


def _rsi_from_history(period: int = 14) -> Optional[float]:
    if len(history) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(len(history) - period + 1, len(history)):
        delta = history[i]["price"] - history[i - 1]["price"]
        if delta > 0:
            gains += delta
        else:
            losses += -delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_like(values: deque) -> float:
    if len(values) == 0:
        return 0.0
    return sum(abs(v) for v in values) / len(values)


def _momentum_positive() -> bool:
    # Default momentum check using base lookback
    if len(_returns) < BASE_MOMENTUM_LOOKBACK:
        return False
    recent = list(_returns)[-BASE_MOMENTUM_LOOKBACK:]
    return sum(recent) > 0


def _adapt_params_by_regime(atr: float):
    """
    Returns a dict of adapted parameters based on current volatility regime.
    - atr < VOL_LOW_THRESH: low vol regime
    - atr > VOL_HIGH_THRESH: high vol regime
    - else: base (mid) regime
    """
    # Start with base
    params = {
        "min_alloc": BASE_MIN_ALLOC,
        "max_alloc": BASE_MAX_ALLOC,
        "hysteresis_pct": BASE_HYSTERESIS_PCT,
        "scale_floor": BASE_SCALE_FLOOR,
        "target_vol": BASE_TARGET_VOL,
        "momentum_lookback": BASE_MOMENTUM_LOOKBACK,
    }
    if atr < VOL_LOW_THRESH:
        params.update(
            {
                "min_alloc": LO_VOL_MIN_ALLOC,
                "max_alloc": LO_VOL_MAX_ALLOC,
                "hysteresis_pct": LO_VOL_HYSTERESIS_PCT,
                "scale_floor": LO_VOL_SCALE_FLOOR,
                "momentum_lookback": LO_VOL_MOMENTUM_LOOKBACK,
            }
        )
    elif atr > VOL_HIGH_THRESH:
        params.update(
            {
                "min_alloc": HI_VOL_MIN_ALLOC,
                "max_alloc": HI_VOL_MAX_ALLOC,
                "hysteresis_pct": HI_VOL_HYSTERESIS_PCT,
                "scale_floor": HI_VOL_SCALE_FLOOR,
                "momentum_lookback": HI_VOL_MOMENTUM_LOOKBACK,
            }
        )
    return params


def _momentum_positive_with_lookback(lookback: int) -> bool:
    if len(_returns) < lookback:
        return False
    recent = list(_returns)[-lookback:]
    return sum(recent) > 0


def _momentum_positive() -> bool:
    if len(_returns) < MOMENTUM_LOOKBACK:
        return False
    recent = list(_returns)[-MOMENTUM_LOOKBACK:]
    return sum(recent) > 0


# ----------------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------------


def make_decision(epoch: int, price: float):
    global _prev_alloc

    # Update state
    history.append({"epoch": epoch, "price": price})
    _prices_short.append(price)
    _prices_long.append(price)
    _update_state(price)
    macd, macd_sig = _update_macd(price)

    # Warmup
    if len(history) < 2 or len(_prices_long) < 50:
        return {"Asset A": 0.5, "Cash": 0.5}

    # Signals
    sma_short = _sma(_prices_short)
    sma_long = _sma(_prices_long)

    # Compute ATR-like vol and adapt parameters for current regime
    atr = _atr_like(_returns)
    regime = _adapt_params_by_regime(atr)

    # Adaptive hysteresis margin and momentum lookback
    margin = regime["hysteresis_pct"] * price
    sma_up = (
        sma_short is not None
        and sma_long is not None
        and (sma_short - sma_long) > margin
    )
    macd_up = macd > macd_sig
    mom_up = _momentum_positive_with_lookback(regime["momentum_lookback"])

    # Risk-on only when all three are bullish
    if sma_up and macd_up and mom_up:
        base_alloc = 0.75
    else:
        base_alloc = 0.25

    # Clamp base allocation with regime-adaptive bounds
    base_alloc = max(regime["min_alloc"], min(regime["max_alloc"], base_alloc))

    # Volatility targeting (ATR-like)
    # Volatility targeting using regime-adaptive target and floor
    vol_scale = regime["target_vol"] / max(atr, 1e-9)
    vol_scale = max(regime["scale_floor"], min(1.0, vol_scale))
    alloc_vol_scaled = 0.5 + (base_alloc - 0.5) * vol_scale

    # Gentle RSI overlay
    rsi = _rsi_from_history(period=14)
    overlay = 0.0
    if rsi is not None:
        if rsi < 35:
            overlay = +0.02
        elif rsi > 65:
            overlay = -0.02
    alloc_with_rsi = alloc_vol_scaled + overlay

    # Clamp and turnover control
    new_alloc = max(regime["min_alloc"], min(regime["max_alloc"], alloc_with_rsi))
    if abs(new_alloc - _prev_alloc) < TURNOVER_THRESHOLD:
        new_alloc = _prev_alloc
    else:
        _prev_alloc = new_alloc

    return {"Asset A": new_alloc, "Cash": 1 - new_alloc}
