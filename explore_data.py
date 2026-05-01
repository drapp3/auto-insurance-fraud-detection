import os

from sqlalchemy import create_engine, text


DB_URL = os.getenv("FRAUD_DB_URL")
if not DB_URL:
    raise RuntimeError("Set FRAUD_DB_URL before running this script.")
engine = create_engine(DB_URL)

print("Exploring the insurance fraud data...\n")

with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM insurance_claims"))
    print(f"Total records: {result.fetchone()[0]:,}")

    result = conn.execute(text("""
        SELECT "FraudFound_P", COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM insurance_claims
        GROUP BY "FraudFound_P"
        ORDER BY "FraudFound_P"
    """))
    print("\nFraud Distribution:")
    for row in result:
        print(f"  {row[0]}: {row[1]:,} cases ({row[2]}%)")

    result = conn.execute(text("""
        SELECT "Make", COUNT(*) as count
        FROM insurance_claims
        GROUP BY "Make"
        ORDER BY count DESC
        LIMIT 5
    """))
    print("\nTop 5 Car Makes:")
    for row in result:
        print(f"  {row[0]}: {row[1]:,} claims")

    result = conn.execute(text("""
        SELECT
            CASE
                WHEN "Age" < 25 THEN 'Under 25'
                WHEN "Age" BETWEEN 25 AND 35 THEN '25-35'
                WHEN "Age" BETWEEN 36 AND 50 THEN '36-50'
                WHEN "Age" BETWEEN 51 AND 65 THEN '51-65'
                ELSE 'Over 65'
            END as age_group,
            COUNT(*) as count
        FROM insurance_claims
        GROUP BY age_group
        ORDER BY age_group
    """))
    print("\nAge Distribution:")
    for row in result:
        print(f"  {row[0]}: {row[1]:,} claims")

    result = conn.execute(text("""
        SELECT "AccidentArea", "FraudFound_P", COUNT(*) as count
        FROM insurance_claims
        GROUP BY "AccidentArea", "FraudFound_P"
        ORDER BY "AccidentArea", "FraudFound_P"
    """))
    print("\nFraud by Accident Area:")
    current_area = None
    for row in result:
        if current_area != row[0]:
            current_area = row[0]
            print(f"\n  {current_area}:")
        print(f"    Fraud {row[1]}: {row[2]:,} cases")

print("\nData exploration complete. Ready to start feature engineering.")
