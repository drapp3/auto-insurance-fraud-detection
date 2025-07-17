import pandas as pd
from sqlalchemy import create_engine, text

# Connection
engine = create_engine('postgresql://postgres:localdev2025@localhost:5432/fraud_detection_db')

# Read the data
print("Loading data...")
df = pd.read_sql("SELECT * FROM insurance_claims", engine)
print(f"Loaded {len(df)} rows")

# Clean the text columns that should be numbers
# Use raw strings (r'...') to avoid escape sequence warnings
# Handle NaN values by filling with 0 before converting
df['AgeOfVehicle_clean'] = df['AgeOfVehicle'].str.extract(r'(\d+)').fillna(0).astype(int)
df['Days_Policy_Accident_clean'] = df['Days_Policy_Accident'].str.extract(r'(\d+)').fillna(0).astype(int)
df['Days_Policy_Claim_clean'] = df['Days_Policy_Claim'].str.extract(r'(\d+)').fillna(0).astype(int)

# Create features
df['claim_delay'] = df['Days_Policy_Claim_clean'] - df['Days_Policy_Accident_clean']
df['is_weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
df['is_young_driver'] = (df['Age'] < 25).astype(int)
df['no_documentation'] = ((df['WitnessPresent'] == 'No') & (df['PoliceReportFiled'] == 'No')).astype(int)

# Save cleaned data
df.to_csv('data/processed/fraud_data_clean.csv', index=False)
print("Data saved to fraud_data_clean.csv")

# Quick fraud analysis
print(f"\nFraud rate: {df['FraudFound_P'].mean():.2%}")
print(f"Average claim delay for fraud: {df[df['FraudFound_P']==1]['claim_delay'].mean():.1f} days")
print(f"Average claim delay for legitimate: {df[df['FraudFound_P']==0]['claim_delay'].mean():.1f} days")