# Leakage-Safe Blending Ensemble for AZN Direction Classification

CQF final project by Yinchi Song. This repository predicts the next executable
daily adjusted open-to-open direction of AstraZeneca ADR (`AZN`) and evaluates
the signal in a long-or-cash strategy with transaction costs.

The model combines Logistic Regression, Random Forest and XGBoost probabilities
with an L2-regularised Logistic Regression blender. All train/blend/test splits
are chronological, and fitted preprocessing is estimated inside the relevant
training folds to reduce look-ahead bias.

## Repository structure

```text
.
|-- notebooks/
|   `-- AZN_Blending_Ensemble.ipynb
|-- src/
|   |-- evaluation.py
|   |-- features.py
|   `-- preprocessing.py
|-- data/                       # cached adjusted OHLCV inputs
|-- results/                    # submitted result tables
|-- reports/
|   `-- CQF_Final_Project_Report.pdf
|-- requirements.txt
|-- requirements-lock.txt
`-- LICENSE
```

## Methodology

- Target: positive adjusted open-to-open return over the next executable daily
  holding period.
- Predicted asset: `AZN`.
- Explanatory markets: `SPY`, `XLV`, `IBB`, `^VIX`, and `GBPUSD=X`.
- Candidate predictors: returns, trend, volatility, volume, correlation and
  cross-market features.
- Split: 60% base-model training, 20% blending, 20% final holdout test.
- Leakage controls: chronological splitting, boundary purging, fold-local
  clipping and feature selection, and a frozen final test specification.
- Strategy: long AZN when probability exceeds the selected cutoff; otherwise
  cash, with explicit turnover costs.

## Reproduce the analysis

Python 3.13 was used for the submitted experiment.

```bash
python -m venv .venv
```

Activate the environment, then install either the portable dependency ranges:

```bash
python -m pip install -r requirements.txt
```

or the exact submitted versions:

```bash
python -m pip install -r requirements-lock.txt
```

Start Jupyter from the repository root and run the notebook from top to bottom:

```bash
jupyter lab notebooks/AZN_Blending_Ensemble.ipynb
```

The notebook downloads current market data from Yahoo Finance and overwrites
the cached CSV files. For an audit of the submitted experiment, use the CSVs
already stored in `data/`. A complete submitted run should produce 238 final
test observations and regenerate the tables in `results/`.

## Important limitations

- This is an academic classification study, not investment advice.
- Results for a single large-cap pharmaceutical ADR do not establish that the
  model generalises to small-cap biotechnology stocks or other asset classes.
- Yahoo Finance may revise adjusted historical data, so a fresh download can
  differ slightly from the cached submission data.
- Hyperparameters and cutoffs must not be retuned using the final holdout set.

## License

Code is released under the MIT License. The report and cached market data may
be subject to separate academic, publisher, or data-provider terms.


