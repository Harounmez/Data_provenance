import pandas as pd
import sys
import sys
import os
import time 
import numpy as np
# Add the path where provenancelib.py is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\harou\Desktop\Dataprovenance_master')))

from provenancelib import Provenance 


# Set input and output paths
input_path = 'use cases\datasets\census.csv'
filename_ext = os.path.basename(input_path)
filename, ext = os.path.splitext(filename_ext)
output_path = 'prov_results'
savepath = os.path.join(output_path, filename)

# Load the dataset
df = pd.read_csv(input_path)

# Assign names to columns
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
df.columns = names

# Initialize Provenance object
p = Provenance()

print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Initialization')

# OPERATION 0: Cleanup names from spaces
prev_df = df.copy()  # Store a copy before transformation
col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'label']
for c in col:
    df[c] = df[c].map(str.strip)

# Capture provenance after cleaning up names
p.capture_data_transformation(prev_df, df)
print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed Operation 0 (Cleanup names from spaces)')

# OPERATION 1: Replace '?' character for NaN value
prev_df = df.copy()  # Store a copy before transformation
df = df.replace('?', np.nan)

# Capture provenance after replacing '?' with NaN
p.capture_data_transformation(prev_df, df)
print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed Operation 1 (Replace ? with NaN)')

# OPERATION 2-3: One-hot encode categorical variables
prev_df = df.copy()  # Store a copy before transformation
col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']

for c in col:
    dummies = pd.get_dummies(df[c], prefix=c)  # Create dummy variables
    df = pd.concat([df, dummies], axis=1)      # Concatenate the dummy columns to the DataFrame
    df = df.drop([c], axis=1)                  # Drop the original categorical column
    
    # Capture provenance after one-hot encoding for each column
    p.capture_vertical_augmentation(prev_df, df)
    prev_df = df.copy()  # Update the previous DataFrame for the next iteration
    print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed One-Hot Encoding for {c}')

# OPERATION 4: Assign binary values to 'sex' and 'label'
prev_df = df.copy()  # Store a copy before transformation
df['sex'] = df['sex'].replace({'Male': 1, 'Female': 0})
df['label'] = df['label'].replace({'<=50K': 0, '>50K': 1})

# Capture provenance after binary assignment
p.capture_data_transformation(prev_df, df)
print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed Operation 4 (Binary assignment for sex and label)')

# OPERATION 5: Drop 'fnlwgt' variable
prev_df = df.copy()  # Store a copy before transformation
df = df.drop(['fnlwgt'], axis=1)

# Capture provenance after dropping 'fnlwgt'
p.capture_vertical_reduction(prev_df, df)
print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Completed Operation 5 (Drop fnlwgt)')

print(f'[{time.strftime("%d/%m-%H:%M:%S")}] Provenance captured and saved successfully.')