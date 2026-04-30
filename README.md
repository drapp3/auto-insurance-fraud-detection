# Auto Insurance Fraud Detection

Machine learning workflow using auto insurance claims data to explore fraud detection, class imbalance, threshold tuning, and business tradeoffs.

The model is tuned for recall because missed fraud is more costly than sending extra claims to manual review. The tradeoff is low precision, which is common in fraud-screening workflows where the model is used as a triage tool rather than an automatic decision-maker.

## Results

- Dataset: 15,420 insurance claims with a 5.99% fraud rate
- Model: Random Forest with SMOTE for class imbalance
- ROC-AUC: 0.673
- Recall: 61% of fraudulent claims caught in the test set
- Precision: 10%, meaning many flagged claims would still need human review
- Scenario estimate: about $98K in potential annual savings under the project assumptions

The savings estimate is a simple scenario calculation based on assumed fraud losses and review costs. It is not a production financial model.

## Workflow

1. Load and inspect claims data
2. Use SQL and Python to explore fraud patterns
3. Engineer claim, timing, location, and demographic risk indicators
4. Train baseline Logistic Regression and Random Forest models
5. Apply SMOTE to address class imbalance
6. Tune the classification threshold around recall and review volume
7. Export charts and model outputs for review

## Key Findings

- Age was the strongest feature in the trained Random Forest model
- Vehicle age and weekend accident timing showed useful signal
- Claims without a police report or witnesses were higher risk
- The optimized threshold caught more fraud but increased false positives

## Project Structure

```text
auto-insurance-fraud-detection/
├── data/
├── models/
├── notebooks/
├── sql/
├── visualizations/
├── explore_data.py
├── load_data.py
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/drapp3/auto-insurance-fraud-detection.git
cd auto-insurance-fraud-detection
pip install -r requirements.txt

createdb fraud_detection_db

python load_data.py
python notebooks/01_data_preparation.py
python notebooks/03_fraud_model_improved.py
python notebooks/04_create_visualizations.py
```

## Notes

This project is focused on the modeling workflow and tradeoff analysis. It is not intended for production fraud decisions without stronger validation, monitoring, and business-specific cost assumptions.

## Contact

- LinkedIn: davisrapp
- Email: drappanalysis@gmail.com
