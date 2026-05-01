import os

import pandas as pd
from sqlalchemy import create_engine, text


DB_URL = os.getenv("FRAUD_DB_URL")
if not DB_URL:
    raise RuntimeError("Set FRAUD_DB_URL before running this script.")


print("Reading CSV file...")
df = pd.read_csv("data/raw/fraud_oracle.csv")
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns in dataset:")
print(df.columns.tolist())

engine = create_engine(DB_URL)

print("\nLoading data to PostgreSQL...")
try:
    df.to_sql("insurance_claims", engine, if_exists="replace", index=False)
    print("Data successfully loaded to PostgreSQL!")

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM insurance_claims"))
        count = result.fetchone()[0]
        print(f"Verified: {count} rows in database")

        result = conn.execute(text("""
            SELECT "FraudFound_P", COUNT(*) as count
            FROM insurance_claims
            GROUP BY "FraudFound_P"
        """))
        print("\nFraud distribution:")
        for row in result:
            print(f"  {row[0]}: {row[1]} cases")

except Exception as e:
    print(f"Error: {e}")
