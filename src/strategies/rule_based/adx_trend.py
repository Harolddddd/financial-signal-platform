from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class ADXTrend(Strategy):
    """Buy when ADX > threshold (strong trend) and +DI > -DI (bullish direction)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 14, adx_threshold: float = 25.0) -> None:
        self.period = period
        self.adx_threshold = adx_threshold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        span = 2 * self.period - 1
        tr_smooth = tr.ewm(span=span, adjust=False).mean()
        plus_dm_s = plus_dm.ewm(span=span, adjust=False).mean()
        minus_dm_s = minus_dm.ewm(span=span, adjust=False).mean()

        safe_tr = tr_smooth.replace(0.0, float("nan"))
        plus_di = 100.0 * plus_dm_s / safe_tr
        minus_di = 100.0 * minus_dm_s / safe_tr

        di_sum = (plus_di + minus_di).replace(0.0, float("nan"))
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(span=span, adjust=False).mean()

        buy = (adx > self.adx_threshold) & (plus_di > minus_di)
        conf_raw = (adx / 100.0).clip(0.0, 1.0) * ((plus_di - minus_di) / 100.0).clip(0.0, 1.0)
        confidence = conf_raw.where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
