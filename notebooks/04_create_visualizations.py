import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load the data and model
df = pd.read_csv('data/processed/fraud_data_clean.csv')
model_data = joblib.load('models/fraud_detector_improved.pkl')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create visualizations directory if it doesn't exist
import os
os.makedirs('visualizations/images', exist_ok=True)

# 1. Fraud Distribution
plt.figure(figsize=(10, 6))
fraud_counts = df['FraudFound_P'].value_counts()
plt.subplot(1, 2, 1)
plt.pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Fraud vs Legitimate Claims')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='AccidentArea', hue='FraudFound_P')
plt.title('Fraud by Accident Area')
plt.legend(['Legitimate', 'Fraud'])
plt.tight_layout()
plt.savefig('visualizations/images/fraud_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Age Analysis
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df_fraud = df[df['FraudFound_P'] == 1]
df_legit = df[df['FraudFound_P'] == 0]
plt.hist([df_legit['Age'], df_fraud['Age']], bins=20, label=['Legitimate', 'Fraud'], alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution by Fraud Status')
plt.legend()

plt.subplot(1, 2, 2)
age_groups = pd.cut(df['Age'], bins=[0, 25, 35, 50, 65, 100], labels=['<25', '25-35', '36-50', '51-65', '65+'])
fraud_by_age = df.groupby([age_groups, 'FraudFound_P']).size().unstack()
fraud_by_age.plot(kind='bar', stacked=False)
plt.title('Claims by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Claims')
plt.legend(['Legitimate', 'Fraud'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizations/images/age_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature Importance
feature_importance = pd.DataFrame({
    'feature': model_data['features'],
    'importance': model_data['model'].feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance Score')
plt.title('Top 15 Most Important Features for Fraud Detection')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Model Performance Visualization
# Create a sample confusion matrix (using the values from your model output)
cm = np.array([[1836, 1063], [72, 113]])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted\nLegitimate', 'Predicted\nFraud'],
            yticklabels=['Actual\nLegitimate', 'Actual\nFraud'])
plt.title('Confusion Matrix - Fraud Detection Model')
plt.tight_layout()
plt.savefig('visualizations/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Business Impact Dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Metric cards
metrics = {
    'Total Claims': 3084,
    'Fraud Caught': 113,
    'Fraud Missed': 72,
    'False Alarms': 1063
}

# Business value
ax1.text(0.5, 0.7, 'Estimated Annual Savings', ha='center', va='center', fontsize=16, weight='bold')
ax1.text(0.5, 0.3, '$98,700', ha='center', va='center', fontsize=36, color='green', weight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Detection rate
detection_rate = 113 / 185 * 100
ax2.pie([detection_rate, 100-detection_rate], labels=['Caught', 'Missed'], 
        autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
ax2.set_title('Fraud Detection Rate', fontsize=14, weight='bold')

# Precision vs Recall
ax3.bar(['Precision', 'Recall'], [0.10, 0.61], color=['blue', 'orange'])
ax3.set_ylim(0, 1)
ax3.set_ylabel('Score')
ax3.set_title('Model Performance Metrics', fontsize=14, weight='bold')
for i, v in enumerate([0.10, 0.61]):
    ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

# Claims processed
ax4.bar(['Total Claims', 'Flagged for Review'], [3084, 113+1063], color=['gray', 'red'])
ax4.set_ylabel('Number of Claims')
ax4.set_title('Claims Processing', fontsize=14, weight='bold')

plt.suptitle('Fraud Detection Model - Business Impact Dashboard', fontsize=18, weight='bold')
plt.tight_layout()
plt.savefig('visualizations/images/business_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations created successfully!")
print("\nFiles created:")
print("- visualizations/images/fraud_distribution.png")
print("- visualizations/images/age_analysis.png") 
print("- visualizations/images/feature_importance.png")
print("- visualizations/images/confusion_matrix.png")
print("- visualizations/images/business_dashboard.png")