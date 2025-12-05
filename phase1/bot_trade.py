from typing import List, Dict

price_history: List[float] = []

sma_window = 60

small_deviation_threshold = 0.005
large_deviation_threshold = 0.015

weight_deep_undervalue = 1.0
weight_mild_undervalue = 0.7
weight_near_fair = 0.3
weight_mild_overvalue = 0.1
weight_deep_overvalue = 0.0


def compute_sma(values):
    return sum(values) / len(values)


def make_decision(epoch: int, current_price: float) -> Dict[str, float]:
    price_history.append(current_price)

    if epoch < sma_window - 1:
        allocation = weight_near_fair
        return {"Asset A": allocation, "Cash": 1.0 - allocation}

    start_index = epoch - sma_window + 1
    end_index = epoch + 1
    recent_prices = price_history[start_index:end_index]

    sma_value = compute_sma(recent_prices)
    deviation = (current_price - sma_value) / sma_value

    if deviation <= -large_deviation_threshold:
        allocation = weight_deep_undervalue
    elif deviation <= -small_deviation_threshold:
        allocation = weight_mild_undervalue
    elif deviation < small_deviation_threshold:
        allocation = weight_near_fair
    elif deviation < large_deviation_threshold:
        allocation = weight_mild_overvalue
    else:
        allocation = weight_deep_overvalue

    return {"Asset A": allocation, "Cash": 1.0 - allocation}
