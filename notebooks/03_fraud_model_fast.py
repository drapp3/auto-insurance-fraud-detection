import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Load cleaned data
print("Loading cleaned data...")
df = pd.read_csv('data/processed/fraud_data_clean.csv')

# Use fewer features for faster training
numeric_features = ['Age', 'AgeOfVehicle_clean', 'claim_delay', 'Deductible', 
                   'Days_Policy_Accident_clean', 'Days_Policy_Claim_clean']
categorical_features = ['AccidentArea', 'Sex', 'PolicyType', 'BasePolicy']

# One-hot encode only important categorical variables
df_encoded = pd.get_dummies(df[categorical_features])

# Prepare features and target
X = pd.concat([df[numeric_features], df_encoded], axis=1)
y = df['FraudFound_P']

print(f"Features shape: {X.shape} (reduced from 55)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest - faster settings
print("\nTraining Random Forest (fast version)...")
rf_model = RandomForestClassifier(
    n_estimators=30,  # Reduced from 100
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',  # Handle imbalance without SMOTE
    max_depth=10  # Limit tree depth for speed
)
rf_model.fit(X_train_scaled, y_train)  # No SMOTE, much faster
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Use probability threshold to catch more fraud
threshold = 0.3  # Lower threshold to catch more fraud
rf_pred = (rf_proba >= threshold).astype(int)

print("\nResults:")
print(classification_report(y_test, rf_pred, target_names=['Legitimate', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.3f}")

# Save the model
model_data = {
    'model': rf_model,
    'scaler': scaler,
    'threshold': threshold,
    'features': X.columns.tolist()
}
joblib.dump(model_data, 'models/fraud_detector_final.pkl')
print("\nModel saved!")