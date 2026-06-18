from pathlib import Path

PARQUET_DIR  = Path("data/features")
REGISTRY_DIR = Path("data/registry")

FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]

CONFIDENCE_THRESHOLD = 0.75
GRADE_COLORS = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c"}
