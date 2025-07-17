-- Feature Engineering for Fraud Detection
-- This query creates new features that help identify fraud patterns

CREATE TABLE fraud_features AS
WITH claim_history AS (
    -- Calculate historical claim patterns per policy
    SELECT 
        "PolicyNumber",
        COUNT(*) as total_claims,
        SUM(CASE WHEN "FraudFound_P" = 1 THEN 1 ELSE 0 END) as previous_frauds,
        AVG("Age") as avg_age
    FROM insurance_claims
    GROUP BY "PolicyNumber"
),
fraud_rates AS (
    -- Calculate fraud rates by different dimensions
    SELECT 
        "Make",
        AVG(CASE WHEN "FraudFound_P" = 1 THEN 1.0 ELSE 0.0 END) as make_fraud_rate
    FROM insurance_claims
    GROUP BY "Make"
)
SELECT 
    ic.*,
    -- Time-based features
    CASE 
        WHEN ic."DayOfWeek" IN ('Saturday', 'Sunday') THEN 1 
        ELSE 0 
    END as is_weekend,
    CASE 
        WHEN ic."DayOfWeekClaimed" IN ('Saturday', 'Sunday') THEN 1 
        ELSE 0 
    END as claimed_weekend,
    
    -- Age-related features
    CASE 
        WHEN ic."Age" < 25 THEN 1 
        ELSE 0 
    END as is_young_driver,
    ic."AgeOfVehicle" * ic."Age" as age_vehicle_interaction,
    
    -- Claim pattern features
    ch.total_claims,
    ch.previous_frauds,
    CASE 
        WHEN ch.total_claims > 3 THEN 1 
        ELSE 0 
    END as multiple_claims,
    
    -- Risk indicators
    CASE 
        WHEN ic."WitnessPresent" = 'No' AND ic."PoliceReportFiled" = 'No' THEN 1 
        ELSE 0 
    END as no_documentation,
    CASE 
        WHEN ic."NumberOfSuppliments" = 'more than 5' THEN 1 
        ELSE 0 
    END as high_supplements,
    
    -- Fraud rates by category
    fr.make_fraud_rate,
    
    -- Deductible analysis
    CAST(REPLACE(REPLACE(ic."Deductible", '$', ''), ',', '') AS INTEGER) as deductible_amount,
    
    -- Days between accident and claim
    ic."Days_Policy_Claim" - ic."Days_Policy_Accident" as claim_delay
    
FROM insurance_claims ic
LEFT JOIN claim_history ch ON ic."PolicyNumber" = ch."PolicyNumber"
LEFT JOIN fraud_rates fr ON ic."Make" = fr."Make";

-- Create indexes for better performance
CREATE INDEX idx_fraud_features_policy ON fraud_features("PolicyNumber");
CREATE INDEX idx_fraud_features_fraud ON fraud_features("FraudFound_P");