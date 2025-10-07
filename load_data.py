import pandas as pd
from sqlalchemy import create_engine, text

# Read the CSV file
print("📖 Reading CSV file...")
df = pd.read_csv('data/raw/fraud_oracle.csv')  # Changed to correct filename!
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

# Show first few rows
print("\n📊 First 5 rows:")
print(df.head())

# Show column names
print("\n📋 Columns in dataset:")
print(df.columns.tolist())

# Create connection (replace with your password)
engine = create_engine('postgresql://postgres:localdev2025@localhost:5432/fraud_detection_db')

# Load to PostgreSQL
print("\n📤 Loading data to PostgreSQL...")
try:
    df.to_sql('insurance_claims', engine, if_exists='replace', index=False)
    print("Data successfully loaded to PostgreSQL!")
    
    # Verify the load
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM insurance_claims"))
        count = result.fetchone()[0]
        print(f"Verified: {count} rows in database")
        
        # Check fraud rate
        result = conn.execute(text("""
            SELECT fraud_reported, COUNT(*) as count 
            FROM insurance_claims 
            GROUP BY fraud_reported
        """))
        print("\nFraud distribution:")
        for row in result:
            print(f"  {row[0]}: {row[1]} cases")
            
except Exception as e:
    print(f"❌ Error: {e}")
