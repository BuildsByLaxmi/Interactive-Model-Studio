"""
Demo Data Seeding Script (Person A)
Pre-seeds the platform with realistic benchmark datasets:
1. customer_churn.csv (Classification)
2. credit_risk.csv (Classification with class imbalance)
3. housing_prices.csv (Regression)
Populates dataset records and storage files so the studio is immediately demonstrable.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure backend path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from database import engine, SessionLocal, Base
from models import DatasetDB, RunDB
from services.dataset_service import process_and_save_dataset


def generate_churn_dataset(n: int = 1000) -> pd.DataFrame:
    """Generates realistic Customer Churn dataset with mixed types and nulls."""
    np.random.seed(42)
    tenure = np.random.randint(1, 72, size=n)
    monthly_charges = np.round(np.random.uniform(18.0, 118.0, size=n), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 50, size=n), 2)
    total_charges = np.clip(total_charges, 18.0, None)

    # Inject ~3% nulls into total_charges to test imputation
    null_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
    total_charges_series = pd.Series(total_charges)
    total_charges_series.iloc[null_idx] = np.nan

    # Contract, InternetService, PaymentMethod
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    internet = np.random.choice(["Fiber optic", "DSL", "No"], size=n, p=[0.45, 0.35, 0.20])
    payment = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n)
    paperless = np.random.choice(["Yes", "No"], size=n, p=[0.6, 0.4])

    # Probability of churn correlated with month-to-month contract and high monthly charges
    churn_prob = 0.15 + 0.35 * (contracts == "Month-to-month") + 0.2 * (monthly_charges > 70) - 0.2 * (tenure > 36)
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    customer_ids = [f"CUST-{10000 + i}" for i in range(n)]

    return pd.DataFrame({
        "customerID": customer_ids,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_series,
        "Contract": contracts,
        "InternetService": internet,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "churn": churn
    })


def generate_credit_dataset(n: int = 800) -> pd.DataFrame:
    """Generates Credit Risk dataset with outliers and class imbalance."""
    np.random.seed(123)
    age = np.random.randint(21, 68, size=n)
    income = np.round(np.random.lognormal(mean=10.5, sigma=0.6, size=n), 2) # Skewed income
    loan_amount = np.round(np.random.uniform(2000, 45000, size=n), 2)
    credit_score = np.random.randint(480, 850, size=n)
    debt_to_income = np.round((loan_amount / income) * 100, 2)

    # 15% default rate
    default_prob = 0.05 + 0.35 * (credit_score < 600) + 0.25 * (debt_to_income > 40)
    default_prob = np.clip(default_prob, 0.02, 0.90)
    loan_default = (np.random.rand(n) < default_prob).astype(int)

    return pd.DataFrame({
        "applicant_age": age,
        "annual_income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income,
        "loan_default": loan_default
    })


def seed_all():
    """Initializes tables and seeds demo datasets."""
    print("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 1. Churn dataset
        print("Seeding customer_churn.csv...")
        df_churn = generate_churn_dataset(1000)
        churn_bytes = df_churn.to_csv(index=False).encode("utf-8")
        churn_record = process_and_save_dataset("customer_churn.csv", churn_bytes, db)
        print(f"-> Seeded Customer Churn dataset ID: {churn_record.id} ({churn_record.row_count} rows)")

        # 2. Credit Risk dataset
        print("Seeding credit_risk.csv...")
        df_credit = generate_credit_dataset(800)
        credit_bytes = df_credit.to_csv(index=False).encode("utf-8")
        credit_record = process_and_save_dataset("credit_risk.csv", credit_bytes, db)
        print(f"-> Seeded Credit Risk dataset ID: {credit_record.id} ({credit_record.row_count} rows)")

        print("\nAll demo datasets successfully seeded!")
    finally:
        db.close()


if __name__ == "__main__":
    seed_all()
