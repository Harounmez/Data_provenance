import pandas as pd
import numpy as np
import time
import os
import sys
# Add the path where provenancelib.py is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\harou\Desktop\Dataprovenance_master')))

from provenancelib import Provenance 

# Set input and output paths
input_path = 'use cases\datasets\german.csv'
output_path = 'prov_results'
savepath = os.path.join(output_path, 'German')

# Load the dataset
df = pd.read_csv(input_path, header=0)

# Initialize the Provenance class
p = Provenance()

print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Initialization')

# OPERATION 0: Translate cryptic values to interpretable form
prev_df = df.copy()  # Store a copy before transformation
df = df.replace({'checking': {'A11': 'check_low', 'A12': 'check_mid', 'A13': 'check_high', 'A14': 'check_none'},
                 'credit_history': {'A30': 'debt_none', 'A31': 'debt_noneBank', 'A32': 'debt_onSchedule', 'A33': 'debt_delay', 'A34': 'debt_critical'},
                 'purpose': {'A40': 'pur_newCar', 'A41': 'pur_usedCar', 'A42': 'pur_furniture', 'A43': 'pur_tv', 'A44': 'pur_appliance', 'A45': 'pur_repairs',
                             'A46': 'pur_education', 'A47': 'pur_vacation', 'A48': 'pur_retraining', 'A49': 'pur_business', 'A410': 'pur_other'},
                 'savings': {'A61': 'sav_small', 'A62': 'sav_medium', 'A63': 'sav_large', 'A64': 'sav_xlarge', 'A65': 'sav_none'},
                 'employment': {'A71': 'emp_unemployed', 'A72': 'emp_lessOne', 'A73': 'emp_lessFour', 'A74': 'emp_lessSeven', 'A75': 'emp_moreSeven'},
                 'other_debtors': {'A101': 'debtor_none', 'A102': 'debtor_coApp', 'A103': 'debtor_guarantor'},
                 'property': {'A121': 'prop_realEstate', 'A122': 'prop_agreement', 'A123': 'prop_car', 'A124': 'prop_none'},
                 'other_inst': {'A141': 'oi_bank', 'A142': 'oi_stores', 'A143': 'oi_none'},
                 'housing': {'A151': 'hous_rent', 'A152': 'hous_own', 'A153': 'hous_free'},
                 'job': {'A171': 'job_unskilledNR', 'A172': 'job_unskilledR', 'A173': 'job_skilled', 'A174': 'job_highSkill'},
                 'phone': {'A191': 0, 'A192': 1},
                 'foreigner': {'A201': 1, 'A202': 0},
                 'label': {2: 0}})

# Capture provenance after feature transformation
p.capture_data_transformation(prev_df, df)
print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed Operation 0 (Translate cryptic values)')

# OPERATION 1: Translate 'personal_status' and 'gender'
prev_df = df.copy()  # Store a copy before transformation
df['status'] = np.where(df.personal_status == 'A91', 'divorced',
                        np.where(df.personal_status == 'A92', 'divorced',
                                 np.where(df.personal_status == 'A93', 'single',
                                          np.where(df.personal_status == 'A95', 'single', 'married'))))

df['gender'] = np.where(df.personal_status == 'A92', 0,
                        np.where(df.personal_status == 'A95', 0, 1))

# Capture provenance after space transformation
p.capture_vertical_augmentation(prev_df, df)
print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed Operation 1 (Translate personal status and gender)')

# OPERATION 2: Drop 'personal_status' column
prev_df = df.copy()  # Store a copy before transformation
df = df.drop(['personal_status'], axis=1)

# Capture provenance after dropping the column
p.capture_vertical_reduction(prev_df, df)
print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed Operation 2 (Drop personal_status column)')

# OPERATION 3-13: One-hot encode categorical columns
prev_df = df.copy()  # Store a copy before transformation
col = ['checking', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'other_inst', 'housing', 'job', 'status']
for c in col:
    dummies = pd.get_dummies(df[c], prefix=c)  # Create dummy variables
    df = pd.concat([df, dummies], axis=1)      # Concatenate the dummy columns to the DataFrame
    df = df.drop([c], axis=1)                  # Drop the original categorical column
    
    # Capture provenance after one-hot encoding for each column
    p.capture_vertical_augmentation(prev_df, df)
    prev_df = df.copy()  # Update the previous DataFrame for the next iteration
    print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed One-Hot Encoding for {c}')

print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Provenance captured and saved successfully.')
