from typing import List, Dict, Optional
import math

price_history: List[float] = []
previous_allocation: Optional[float] = None
highest_price_seen: Optional[float] = None

fast_trend_window = 3
medium_trend_window = 15
slow_trend_window = 50
breakout_window = 30
volatility_window = 10
zscore_window = 10

target_annual_volatility = 0.90
trading_days_per_year = 252
target_daily_volatility = target_annual_volatility / math.sqrt(trading_days_per_year)

minimum_volatility_scaler = 0.35
maximum_volatility_scaler = 2.0

minimum_allocation_change = 0.06


def compute_simple_moving_average(values: List[float]) -> float:
    return sum(values) / len(values)


def compute_standard_deviation(values: List[float]) -> float:
    mean_value = compute_simple_moving_average(values)
    squared_differences = [(value - mean_value) ** 2 for value in values]
    variance = sum(squared_differences) / len(values)
    if variance <= 0:
        return 0.0
    return math.sqrt(variance)


def compute_trend_metrics() -> Dict[str, float]:
    fast_window_prices = price_history[-fast_trend_window:]
    medium_window_prices = price_history[-medium_trend_window:]
    slow_window_prices = price_history[-slow_trend_window:]
    fast_moving_average = compute_simple_moving_average(fast_window_prices)
    medium_moving_average = compute_simple_moving_average(medium_window_prices)
    slow_moving_average = compute_simple_moving_average(slow_window_prices)
    if slow_moving_average == 0:
        return {
            "fast_trend_strength": 0.0,
            "medium_trend_strength": 0.0,
            "fast_moving_average": fast_moving_average,
            "medium_moving_average": medium_moving_average,
            "slow_moving_average": slow_moving_average,
        }
    fast_trend_strength = (fast_moving_average - slow_moving_average) / slow_moving_average
    medium_trend_strength = (medium_moving_average - slow_moving_average) / slow_moving_average
    return {
        "fast_trend_strength": fast_trend_strength,
        "medium_trend_strength": medium_trend_strength,
        "fast_moving_average": fast_moving_average,
        "medium_moving_average": medium_moving_average,
        "slow_moving_average": slow_moving_average,
    }


def compute_zscore_relative_to_fast_ma(current_price: float, fast_moving_average: float) -> float:
    if len(price_history) < max(zscore_window, fast_trend_window):
        return 0.0
    recent_prices = price_history[-zscore_window:]
    standard_deviation = compute_standard_deviation(recent_prices)
    if standard_deviation == 0:
        return 0.0
    return (current_price - fast_moving_average) / standard_deviation


def compute_volatility_scaler() -> float:
    if len(price_history) < volatility_window + 1:
        return 1.0
    recent_prices = price_history[-(volatility_window + 1):]
    returns = [
        recent_prices[index + 1] / recent_prices[index] - 1.0
        for index in range(volatility_window)
    ]
    mean_return = compute_simple_moving_average(returns)
    squared_differences = [(value - mean_return) ** 2 for value in returns]
    variance = sum(squared_differences) / len(returns)
    if variance <= 0:
        return 1.0
    realized_daily_volatility = math.sqrt(variance)
    if realized_daily_volatility == 0:
        return 1.0
    raw_scaler = target_daily_volatility / realized_daily_volatility
    if raw_scaler < minimum_volatility_scaler:
        return minimum_volatility_scaler
    if raw_scaler > maximum_volatility_scaler:
        return maximum_volatility_scaler
    return raw_scaler


def compute_breakout_factor(current_price: float) -> float:
    if len(price_history) < breakout_window:
        return 0.0
    recent_prices = price_history[-breakout_window:]
    highest_recent_price = max(recent_prices)
    if highest_recent_price == 0:
        return 0.0
    distance_from_high = (current_price - highest_recent_price) / highest_recent_price
    if distance_from_high >= 0.0:
        return 1.0
    if distance_from_high >= -0.01:
        return 0.7
    return 0.0


def apply_drawdown_overlay(allocation: float, current_price: float) -> float:
    global highest_price_seen
    if highest_price_seen is None or current_price > highest_price_seen:
        highest_price_seen = current_price
    if highest_price_seen is None or highest_price_seen <= 0:
        return allocation
    drawdown = current_price / highest_price_seen - 1.0
    if drawdown <= -0.45:
        return 0.0
    if drawdown <= -0.35 and allocation > 0.4:
        return 0.4
    return allocation


def apply_hysteresis(allocation: float) -> float:
    global previous_allocation
    if previous_allocation is None:
        previous_allocation = allocation
        return allocation
    if abs(allocation - previous_allocation) < minimum_allocation_change:
        allocation = previous_allocation
    else:
        previous_allocation = allocation
    return allocation


def compute_regime_based_allocation(current_price: float) -> float:
    trend_metrics = compute_trend_metrics()
    fast_trend_strength = trend_metrics["fast_trend_strength"]
    medium_trend_strength = trend_metrics["medium_trend_strength"]
    fast_moving_average = trend_metrics["fast_moving_average"]
    slow_moving_average = trend_metrics["slow_moving_average"]

    if slow_moving_average == 0:
        return 0.7

    price_vs_slow = (current_price - slow_moving_average) / slow_moving_average
    is_strong_uptrend = (
        fast_moving_average > slow_moving_average
        and fast_trend_strength > 0.01
        and medium_trend_strength > 0.004
    )
    is_bearish_regime = (
        fast_moving_average < slow_moving_average
        and price_vs_slow < -0.01
    )
    zscore_fast = compute_zscore_relative_to_fast_ma(current_price, fast_moving_average)
    breakout_factor = compute_breakout_factor(current_price)

    if is_strong_uptrend and price_vs_slow > 0.01:
        base_allocation = 0.9
        if breakout_factor >= 1.0:
            base_allocation = 1.0
        if zscore_fast <= -0.5:
            base_allocation = 1.0
        if zscore_fast >= 1.5:
            base_allocation = 0.8
        return base_allocation

    if price_vs_slow > 0.0:
        if fast_trend_strength > 0.005 or medium_trend_strength > 0.003:
            base_allocation = 0.8
        else:
            base_allocation = 0.6
        if zscore_fast <= -0.7:
            base_allocation = min(1.0, base_allocation + 0.2)
        return base_allocation

    if is_bearish_regime:
        if fast_trend_strength < -0.03:
            return 0.0
        if fast_trend_strength < -0.015:
            return 0.2
        return 0.3

    if fast_trend_strength >= 0.0:
        return 0.5
    return 0.3


def make_decision(epoch: int, current_price: float) -> Dict[str, float]:
    global highest_price_seen

    price_history.append(current_price)

    if highest_price_seen is None or current_price > highest_price_seen:
        highest_price_seen = current_price

    warmup_period = max(slow_trend_window, volatility_window + 1, zscore_window + 1)
    if epoch < warmup_period:
        initial_allocation = 0.7
        final_allocation = apply_hysteresis(initial_allocation)
        return {"Asset B": final_allocation, "Cash": 1.0 - final_allocation}

    regime_allocation = compute_regime_based_allocation(current_price)
    volatility_scaler = compute_volatility_scaler()

    if regime_allocation > 0.7 and volatility_scaler < 1.0:
        effective_scaler = max(0.9, volatility_scaler)
    else:
        effective_scaler = volatility_scaler

    scaled_allocation = regime_allocation * effective_scaler

    if scaled_allocation < 0.0:
        scaled_allocation = 0.0
    if scaled_allocation > 1.0:
        scaled_allocation = 1.0

    allocation_with_drawdown = apply_drawdown_overlay(scaled_allocation, current_price)
    final_allocation = apply_hysteresis(allocation_with_drawdown)

    return {"Asset B": final_allocation, "Cash": 1.0 - final_allocation}
