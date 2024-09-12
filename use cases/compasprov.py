import pandas as pd
import sys
import sys
import os

# Add the path where provenancelib.py is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\harou\Desktop\Dataprovenance_master')))

from provenancelib import Provenance 
# Load the dataset
df = pd.read_csv('use cases\datasets\compas.csv', header=0)

# Initialize the Provenance class
p = Provenance()

# OPERATION 0: Select relevant columns
prev_df = df.copy()  # Store a copy before transformation
df = df[['age', 'c_charge_degree', 'race', 'sex', 'priors_count', 'days_b_screening_arrest', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
p.capture_vertical_reduction(prev_df, df)

# OPERATION 1: Remove missing values
prev_df = df.copy()  # Store a copy before transformation
df = df.dropna()
p.capture_vertical_reduction(prev_df, df)

# OPERATION 2: Make race binary
prev_df = df.copy()  # Store a copy before transformation
df.race = [0 if r != 'Caucasian' else 1 for r in df.race]
p.capture_data_transformation(prev_df, df)

# OPERATION 3: Make two_year_recid the label
prev_df = df.copy()  # Store a copy before transformation
df = df.rename({'two_year_recid': 'label'}, axis=1)
df.label = [0 if l == 1 else 1 for l in df.label]
p.capture_data_transformation(prev_df, df)

# OPERATION 4: Convert jailtime to days
prev_df = df.copy()  # Store a copy before transformation
df['jailtime'] = (pd.to_datetime(df.c_jail_out) - pd.to_datetime(df.c_jail_in)).dt.days
p.capture_vertical_augmentation(prev_df, df)

# OPERATION 5: Drop jail in and out dates
prev_df = df.copy()  # Store a copy before transformation
df = df.drop(['c_jail_in', 'c_jail_out'], axis=1)
p.capture_vertical_reduction(prev_df, df)

# OPERATION 6: Convert charge degree to binary
prev_df = df.copy()  # Store a copy before transformation
df.c_charge_degree = [0 if s == 'M' else 1 for s in df.c_charge_degree]
p.capture_data_transformation(prev_df, df)

print('Provenance captured and saved.')