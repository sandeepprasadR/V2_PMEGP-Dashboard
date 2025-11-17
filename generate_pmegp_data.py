"""
PMEGP Data Generator - Generate Realistic Sample Data
=====================================================

This script generates realistic random data matching the actual PMEGP
dataset structure with all the fields shown in the PMEGP division dataset.

Fields included:
- SR_NODD, CURRENT_STATUS, OFF_NAME, AGENCY_TYPE, STATE_NM, APP_ID, APP_NAME
- BENF_TYPE_DESC, GENDER, BENF_CATEGORY_DESC, BENF_SPECIAL_CD, DOB
- MOBILE, UNIT_ADDR, UNIT_TALUK_BLOCK, DISTRICT_NAME, IND_TYPE
- ACTIVITY_NAME, PROD_DESC, IFSC_CODE, BANK_NAME, BANK_ADDRESS, BANK_F_DATE
- PROJ_COST, TOT_SAMC_FB, CGTSI, IST_LOAN_REL, MM_CLAIM_DT, MM_CLAIM_AMT
- MM_REL_DT, MM_REL_AMT, EDP_TRG_NM, EDP_CERT_DT, CERT_TYPE, ONLINE_PROMO
- TEL_NO, E_MAIL, IND_GRP_LONG_DESC, APR_YR

Author: PMEGP Dashboard Development
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration
NUM_RECORDS = 5000
OUTPUT_FILE = '/Users/apple/Documents/0. MSME project/V2_PMEGP_Dashboard/PMEGP_Generated_Data_2023-26.csv'

# Fixed lists for realistic data
STATES = ['HIMACHAL PRADESH', 'UTTAR PRADESH', 'MAHARASHTRA', 'KARNATAKA', 
          'TAMIL NADU', 'RAJASTHAN', 'BIHAR', 'WEST BENGAL', 'PUNJAB', 'HARYANA',
          'ASSAM', 'ODISHA', 'TELANGANA', 'ANDHRA PRADESH', 'KERALA', 'JHARKHAND',
          'CHHATTISGARH', 'MADHYA PRADESH', 'UTTARAKHAND', 'JAMMU & KASHMIR']

AGENCIES = ['DIC', 'KVIC', 'KVIB', 'Ministry', 'State']

ACTIVITY_TYPES = [
    'Retail Trading', 'Service Business', 'Manufacturing', 'Food Processing',
    'Textile Unit', 'Video & Photo Studio', 'Handicraft', 'IT Services',
    'Education Center', 'Health Care', 'E-commerce', 'Tourism Services'
]

GENDERS = ['Male', 'Female']

CATEGORIES = ['General', 'SC', 'ST', 'OBC', 'Scheduled Caste', 'Scheduled Tribe']

STATUSES = ['Operational', 'Closed', 'Non-Operational']

BANKS = ['HDFC Bank', 'ICICI Bank', 'SBI', 'PNB', 'Axis Bank', 'Kotak Bank',
         'Canara Bank', 'Union Bank', 'Bank of Baroda', 'IndusInd Bank']

def generate_pmegp_data(num_records):
    """Generate realistic PMEGP dataset"""
    
    print(f"Generating {num_records} PMEGP records...")
    
    data = {
        'SR_NODD': range(1, num_records + 1),
        'CURRENT_STATUS': [random.choice(STATUSES) for _ in range(num_records)],
        'OFF_NAME': [f'Office_{random.randint(1, 100)}' for _ in range(num_records)],
        'AGENCY_TYPE': [random.choice(AGENCIES) for _ in range(num_records)],
        'STATE_NM': [random.choice(STATES) for _ in range(num_records)],
        'APP_ID': [f'APP{datetime.now().year}{random.randint(100000, 999999)}' 
                   for _ in range(num_records)],
        'APP_NAME': [f'Applicant_{i}' for i in range(1, num_records + 1)],
        'BENF_TYPE_DESC': ['Individual'] * num_records,
        'GENDER': [random.choice(GENDERS) for _ in range(num_records)],
        'BENF_CATEGORY_DESC': [random.choice(CATEGORIES) for _ in range(num_records)],
        'BENF_SPECIAL_CD': [random.choice(['NA', 'PWD', 'EX-SERVICEMAN']) for _ in range(num_records)],
        'DOB': [datetime(1960, 1, 1) + timedelta(days=random.randint(0, 20000)) 
                for _ in range(num_records)],
        'MOBILE': [f'98{random.randint(10000000, 99999999)}' for _ in range(num_records)],
        'UNIT_ADDR': [f'Address_{random.randint(1, 500)}' for _ in range(num_records)],
        'UNIT_TALUK_BLOCK': [f'Block_{random.choice(["A", "B", "C", "D"])}' 
                             for _ in range(num_records)],
        'DISTRICT_NAME': [f'District_{random.randint(1, 50)}' for _ in range(num_records)],
        'IND_TYPE': [random.choice(['Rural', 'Urban']) for _ in range(num_records)],
        'ACTIVITY_NAME': [random.choice(ACTIVITY_TYPES) for _ in range(num_records)],
        'PROD_DESC': [f'Product_{random.randint(1, 100)}' for _ in range(num_records)],
        'IFSC_CODE': [f'IFSC{random.randint(100000, 999999)}' for _ in range(num_records)],
        'BANK_NAME': [random.choice(BANKS) for _ in range(num_records)],
        'BANK_ADDRESS': [f'Bank_Address_{random.randint(1, 200)}' for _ in range(num_records)],
        'BANK_F_DATE': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)) 
                        for _ in range(num_records)],
        'PROJ_COST': [random.randint(100000, 2500000) for _ in range(num_records)],
        'TOT_SAMC_FB': [random.randint(50000, 1500000) for _ in range(num_records)],
        'CGTSI': [random.randint(10000, 500000) for _ in range(num_records)],
        'IST_LOAN_REL': [random.choice(['Yes', 'No', 'Partial']) for _ in range(num_records)],
        'MM_CLAIM_DT': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)) 
                        for _ in range(num_records)],
        'MM_CLAIM_AMT': [random.randint(100000, 2000000) for _ in range(num_records)],
        'MM_REL_DT': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)) 
                      for _ in range(num_records)],
        'MM_REL_AMT': [random.randint(50000, 1500000) for _ in range(num_records)],
        'EDP_TRG_NM': [f'Training_{random.randint(1, 50)}' for _ in range(num_records)],
        'EDP_CERT_DT': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)) 
                        for _ in range(num_records)],
        'CERT_TYPE': [random.choice(['A', 'B', 'C', 'NA']) for _ in range(num_records)],
        'ONLINE_PROMO': [random.choice(['Yes', 'No']) for _ in range(num_records)],
        'TEL_NO': [f'0{random.randint(11, 40)}{random.randint(1000000, 9999999)}' 
                   for _ in range(num_records)],
        'E_MAIL': [f'applicant_{i}@email.com' for i in range(1, num_records + 1)],
        'IND_GRP_LONG_DESC': [random.choice(['Services', 'Manufacturing', 'Retail', 'Agro-based']) 
                              for _ in range(num_records)],
        'APR_YR': [random.choice(['2023-24', '2024-25', '2025-26']) for _ in range(num_records)],
        'ANNUAL_TURNOVER': [random.randint(200000, 5000000) for _ in range(num_records)],
        'SUSTAINABILITY_SCORE': [random.uniform(0, 100) for _ in range(num_records)],
        'OPERATIONAL_STATUS': [random.choice(['Operational', 'Closed', 'Non-Operational']) 
                               for _ in range(num_records)],
        'EMPLOYMENT_AT_SETUP': [random.randint(1, 20) for _ in range(num_records)],
        'MARGIN_MONEY_SUBSIDY_RS': [random.randint(100000, 1500000) for _ in range(num_records)],
    }
    
    df = pd.DataFrame(data)
    
    # Convert date columns to string format
    date_columns = ['DOB', 'BANK_F_DATE', 'MM_CLAIM_DT', 'MM_REL_DT', 'EDP_CERT_DT']
    for col in date_columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Successfully generated {num_records} records")
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print(f"✓ File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"✓ Columns: {len(df.columns)}")
    print(f"\nSample data preview:")
    print(df.head(10))
    
    return df

if __name__ == "__main__":
    print("="*80)
    print("PMEGP DATA GENERATOR")
    print("="*80)
    print()
    
    # Generate data
    df = generate_pmegp_data(NUM_RECORDS)
    
    print()
    print("="*80)
    print("Data generation complete!")
    print("="*80)
    print()
    print("Column names:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
