"""Classification and long/cash backtest evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)

TRADING_DAYS = 252


def classification_metrics(y_true, probability, cutoff=0.5):
    """Return threshold-free and threshold-dependent classification metrics."""
    y = np.asarray(y_true)
    p = np.asarray(probability)
    pred = (p >= cutoff).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "positive_prediction_rate": pred.mean(),
    }


def strategy_returns(asset_return, probability=None, cutoff=0.5,
                     one_way_cost_bps=5.0, always_long=False):
    """Construct aligned gross/net returns for a long/cash strategy."""
    returns = pd.Series(asset_return).astype(float)
    if always_long:
        position = pd.Series(1.0, index=returns.index)
    else:
        position = (pd.Series(probability, index=returns.index) >= cutoff).astype(float)
    turnover = position.diff().abs()
    turnover.iloc[0] = position.iloc[0]
    gross = position * returns
    cost = turnover * one_way_cost_bps / 10_000
    net = gross - cost
    return pd.DataFrame({"asset_return": returns, "position": position,
                         "turnover": turnover, "cost": cost,
                         "gross_return": gross, "net_return": net})


def performance_metrics(result):
    """Calculate annualised and path-dependent strategy statistics."""
    net = result["net_return"]
    n = len(net)
    wealth = (1 + net).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    ann_return = wealth.iloc[-1] ** (TRADING_DAYS / n) - 1
    ann_vol = net.std(ddof=1) * np.sqrt(TRADING_DAYS)
    downside = net[net < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    invested = result["position"] > 0
    return {
        "cumulative_return": wealth.iloc[-1] - 1,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "sharpe": net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS) if net.std(ddof=1) > 0 else np.nan,
        "sortino": net.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        "maximum_drawdown": drawdown.min(),
        "hit_rate_when_invested": (result.loc[invested, "asset_return"] > 0).mean() if invested.any() else np.nan,
        "exposure": result["position"].mean(),
        "position_changes": int(result["turnover"].sum()),
        "turnover": result["turnover"].sum(),
        "total_transaction_cost": result["cost"].sum(),
    }


def bootstrap_auc_interval(y_true, probability, n_bootstrap=2000,
                           confidence=0.95, random_state=42):
    """Percentile bootstrap interval for ROC-AUC, skipping one-class draws."""
    y = np.asarray(y_true)
    p = np.asarray(probability)
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            scores.append(roc_auc_score(y[idx], p[idx]))
    alpha = 1 - confidence
    return np.quantile(scores, [alpha / 2, 1 - alpha / 2])

