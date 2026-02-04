# Auto Insurance Fraud Detection System

## Overview
Machine learning system that identifies fraudulent auto insurance claims with 61% detection rate, potentially saving $98,700 annually for insurers while minimizing false positives.

## Business Impact
- $98,700 estimated annual savings
- 61% fraud detection rate (113 out of 185 fraudulent claims caught)
- 10% precision (acceptable trade-off for high recall in fraud detection)
- 1,176 claims flagged for review out of 3,084 total

## Project Structure
```
auto-insurance-fraud-detection/
├── data/                      # Data files
├── models/                    # Trained models
├── notebooks/                 # Analysis scripts
├── sql/                       # SQL feature engineering
├── visualizations/           # Generated charts
└── README.md                 # This file
```
## Methodology
1. Data Analysis

- Analyzed 15,420 insurance claims with 5.99% fraud rate
- Identified key fraud indicators through SQL queries and Python analysis
- Found urban areas have higher fraud rates (790 vs 133 rural cases)

2. Feature Engineering

- Created risk indicators using PostgreSQL window functions
- Engineered time-based features (weekend claims, claim delays)
- Built demographic risk profiles

3. Model Development

- Tested Logistic Regression and Random Forest algorithms
- Used SMOTE to handle class imbalance
- Optimized threshold for business impact vs accuracy

4. Results

- Random Forest model achieved 0.673 ROC-AUC
- Optimized for high recall to catch more fraud
- Trade-off: More false positives but fewer missed frauds

## Key Insights

- Age is the strongest predictor (49.3% feature importance)
- Vehicle age correlates with fraud risk
- Weekend accidents show higher fraud probability
- Claims with no police report or witnesses are high risk

## Technical Skills Demonstrated

- SQL: Complex queries, CTEs, window functions
- Python: pandas, scikit-learn, feature engineering
- Machine Learning: Classification, imbalanced data handling, threshold optimization
- Business Analysis: ROI calculation, metric interpretation

## Installation & Usage
```bash
# Clone repository
git clone https://github.com/yourusername/auto-insurance-fraud-detection.git

# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL database
createdb fraud_detection_db

# Run analysis pipeline
python load_data.py
python notebooks/01_data_preparation.py
python notebooks/03_fraud_model_improved.py
python notebooks/04_create_visualizations.py
```
## Future Improvements

- Implement real-time scoring API
- Add more external data sources (weather, economic indicators)
- Test deep learning approaches
- Develop explainable AI dashboard

## Contact

- LinkedIn: davisrapp
- Email: drappanalysis@gmail.com
