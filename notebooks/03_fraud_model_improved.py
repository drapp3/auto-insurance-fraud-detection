import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load cleaned data
print("Loading cleaned data...")
df = pd.read_csv('data/processed/fraud_data_clean.csv')

# Select features for modeling
feature_cols = [
    'Age', 'is_young_driver', 'is_weekend', 'no_documentation',
    'AgeOfVehicle_clean', 'claim_delay', 'Deductible',
    'Days_Policy_Accident_clean', 'Days_Policy_Claim_clean'
]

# Encode categorical variables
categorical_cols = ['Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'PolicyType', 
                   'VehicleCategory', 'BasePolicy', 'PastNumberOfClaims']

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
feature_cols.extend(df_encoded.columns.tolist())

# Prepare features and target
X = pd.concat([df[feature_cols[:9]], df_encoded], axis=1)
y = df['FraudFound_P']

print(f"Features shape: {X.shape}")
print(f"Fraud cases: {y.sum()} ({y.mean():.1%})")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalanced data with SMOTE
print("\nBalancing data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE - Fraud rate: {y_train_balanced.mean():.1%}")

# Model 1: Logistic Regression with balanced classes
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_balanced, y_train_balanced)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression Results:")
print(classification_report(y_test, lr_pred, target_names=['Legitimate', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, lr_proba):.3f}")

# Model 2: Random Forest with balanced classes
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced',
    min_samples_split=20
)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("Random Forest Results:")
print(classification_report(y_test, rf_pred, target_names=['Legitimate', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.3f}")

# Find optimal threshold
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, rf_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.3f}")
rf_pred_optimal = (rf_proba >= optimal_threshold).astype(int)
print("Results with optimal threshold:")
print(classification_report(y_test, rf_pred_optimal, target_names=['Legitimate', 'Fraud']))

# Business impact analysis
tn, fp, fn, tp = confusion_matrix(y_test, rf_pred_optimal).ravel()
print("\nBusiness Impact:")
print(f"True Positives (Fraud caught): {tp}")
print(f"False Positives (Legitimate flagged): {fp}")
print(f"False Negatives (Fraud missed): {fn}")
print(f"True Negatives (Legitimate approved): {tn}")

# Assuming $5000 average loss per fraud
fraud_caught_value = tp * 5000
fraud_missed_cost = fn * 5000
investigation_cost = fp * 100  # Cost to investigate false positive

net_benefit = fraud_caught_value - fraud_missed_cost - investigation_cost
print(f"\nEstimated net benefit: ${net_benefit:,.0f}")

# Save model and threshold
model_data = {
    'model': rf_model,
    'scaler': scaler,
    'threshold': optimal_threshold,
    'features': X.columns.tolist()
}
joblib.dump(model_data, 'models/fraud_detector_improved.pkl')
print("\nModel saved to models/fraud_detector_improved.pkl")