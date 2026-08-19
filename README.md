# Loan Approval Modeling

A cost-sensitive machine-learning project for historical loan approval decisions. The analysis follows the CRISP-DM framework from business understanding through model evaluation and implementation recommendations.

## Project overview

The notebook compares five classification model families and selects an interpretable logistic-regression pipeline. It includes:

- Data-quality assessment and exploratory visualizations
- Numeric, nominal, and ordinal preprocessing with scikit-learn pipelines
- Missing-value handling and feature engineering
- Stratified cross-validation and hyperparameter tuning
- A custom business-cost metric using $50,000 per false approval and $8,000 per false denial
- Out-of-fold decision-threshold selection
- Feature interpretation and segment-level performance checks

## Results

On 4,000 untouched test applications, the final model achieved:

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.995 |
| PR-AUC | 0.984 |
| Precision | 97.7% |
| Recall | 80.3% |
| Cost-sensitive threshold | 0.845 |
| Estimated cost reduction vs. always denying | 68.6% |

These results measure agreement with historical approval labels under fixed average costs. They do not establish future default risk or causal creditworthiness.

## Repository contents

- `financial_loan_risk.ipynb` — complete analysis and modeling workflow
- `financial_loan_data.csv` — dataset used by the notebook
- `Loan.csv` — accompanying source dataset
- `ml-env.yml` — reproducible Conda environment

## Environment setup

Create and activate the project environment:

```bash
conda env create -f ml-env.yml
conda activate ml-env
```

Register the Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name ml-env --display-name "Python (ml-env)"
```

Start Jupyter from the repository root:

```bash
jupyter lab financial_loan_risk.ipynb
```

Select the **Python (ml-env)** kernel. The notebook reads `financial_loan_data.csv` from the repository root.

## Limitations

The approval label reflects historical policy rather than observed repayment outcomes, which creates selective-label bias. The analysis also uses fixed average error costs and a random train/test split. Before operational use, the project should be validated on later application cohorts, calibrated against observed defaults, reviewed for feature availability and leakage, and audited with legally appropriate protected-class data.
