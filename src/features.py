"""Leakage-safe candidate feature generation for the AZN project."""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

TRADING_DAYS = 252


def build_candidate_features(data: pd.DataFrame, external_names: list[str]) -> pd.DataFrame:
    """Create time-t predictors using observations available at or before time t."""
    f = pd.DataFrame(index=data.index)
    close, open_ = data["AZN_Close"], data["AZN_Open"]
    high, low, volume = data["AZN_High"], data["AZN_Low"], data["AZN_Volume"]
    log_return = np.log(close).diff()

    for window in [1, 2, 5, 10, 20, 63]:
        f[f"azn_return_{window}d"] = close.pct_change(window)
    for window in [5, 10, 20, 50]:
        f[f"azn_close_to_sma_{window}"] = close / close.rolling(window).mean() - 1
        f[f"azn_close_to_ema_{window}"] = close / close.ewm(span=window, adjust=False).mean() - 1
    f["azn_sma_5_to_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    f["azn_sma_20_to_50"] = close.rolling(20).mean() / close.rolling(50).mean() - 1
    f["azn_overnight_gap"] = open_ / close.shift(1) - 1
    f["azn_intraday_return"] = close / open_ - 1
    f["azn_drawdown_63"] = close / close.rolling(63).max() - 1

    f["azn_rsi_14"] = RSIIndicator(close, window=14).rsi() / 100
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    f["azn_macd_scaled"] = macd.macd() / close
    f["azn_macd_signal_scaled"] = macd.macd_signal() / close
    f["azn_macd_hist_scaled"] = macd.macd_diff() / close
    bb = BollingerBands(close, window=20, window_dev=2)
    f["azn_bb_position"] = bb.bollinger_pband()
    f["azn_bb_width"] = bb.bollinger_wband() / 100
    f["azn_atr_14_scaled"] = AverageTrueRange(high, low, close, window=14).average_true_range() / close
    f["azn_adx_14"] = ADXIndicator(high, low, close, window=14).adx() / 100

    for window in [5, 10, 20]:
        f[f"azn_realized_vol_{window}"] = log_return.rolling(window).std() * np.sqrt(TRADING_DAYS)
    parkinson = np.log(high / low).pow(2) / (4 * np.log(2))
    gk = 0.5 * np.log(high / low).pow(2) - (2 * np.log(2) - 1) * np.log(close / open_).pow(2)
    f["azn_parkinson_vol_20"] = np.sqrt(parkinson.rolling(20).mean() * TRADING_DAYS)
    f["azn_garman_klass_vol_20"] = np.sqrt(gk.clip(lower=0).rolling(20).mean() * TRADING_DAYS)
    f["azn_skew_20"] = log_return.rolling(20).skew()
    f["azn_kurtosis_20"] = log_return.rolling(20).kurt()
    f["azn_downside_vol_20"] = log_return.where(log_return < 0, 0).rolling(20).std() * np.sqrt(TRADING_DAYS)

    log_volume = np.log1p(volume)
    f["azn_volume_change_1d"] = volume.pct_change()
    f["azn_volume_change_5d"] = volume.pct_change(5)
    f["azn_volume_z_20"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    f["azn_obv_change_5d"] = obv.diff(5) / volume.rolling(20).mean()
    f["azn_return_volume_interaction"] = log_return * f["azn_volume_z_20"]

    for name in external_names:
        ext = data[f"{name}_Close"]
        for window in [1, 5, 20]:
            f[f"{name.lower()}_return_{window}d"] = ext.pct_change(window)
        f[f"{name.lower()}_vol_20"] = np.log(ext).diff().rolling(20).std() * np.sqrt(TRADING_DAYS)
    f["azn_minus_spy_5d"] = f["azn_return_5d"] - f["spy_return_5d"]
    f["azn_minus_xlv_5d"] = f["azn_return_5d"] - f["xlv_return_5d"]
    f["azn_xlv_corr_20"] = log_return.rolling(20).corr(np.log(data["XLV_Close"]).diff())
    f["azn_spy_corr_20"] = log_return.rolling(20).corr(np.log(data["SPY_Close"]).diff())
    return f.replace([np.inf, -np.inf], np.nan)

