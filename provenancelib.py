import numpy as np
import pandas as pd
import time
import os
from scipy.sparse import csr_matrix, save_npz

class Provenance:
    def __init__(self):
        self.provenance_data = {}
        self.step_counter = 0  # Initialize step counter

    def save_sparse_tensor(self, tensor, operation_name):
        """Save sparse tensor as binary file."""
        # Increment the step counter
        self.step_counter += 1

        # Create a directory for provenance data if it doesn't exist
        if not os.path.exists('provenance_data'):
            os.makedirs('provenance_data')

        # Define the file name with step number and operation name
        file_name = f'provenance_data/step_{self.step_counter}_{operation_name}.npz'

        # Save the sparse tensor in .npz format
        save_npz(file_name, tensor)
        print(f"Sparse Provenance tensor saved as {file_name}")

    def timed_capture(func):
        """Decorator to time function execution."""
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            print(f"{func.__name__} took {duration:.4f} seconds")
            return result
        return wrapper

    @timed_capture
    def capture_vertical_reduction(self, din, dout):
        n, m = len(din), len(dout)

        # Create a sparse identity matrix for vertical reduction
        sparse_tensor = csr_matrix(np.eye(m, n, dtype=int))

        # Save the sparse tensor
        self.save_sparse_tensor(sparse_tensor, "vertical_reduction")

        self.provenance_tensor = sparse_tensor
        return sparse_tensor

    @timed_capture
    def capture_horizontal_reduction(self, din, dout):
        n, m = len(din), len(dout)

        # Create a sparse identity matrix for horizontal reduction
        sparse_tensor = csr_matrix(np.eye(m, n, dtype=int))

        # Save the sparse tensor
        self.save_sparse_tensor(sparse_tensor, "horizontal_reduction")

        self.provenance_tensor = sparse_tensor
        return sparse_tensor

    @timed_capture
    def capture_data_transformation(self, din, dout):
        n, m = len(din), len(dout)

        # Create a sparse identity matrix for data transformation
        sparse_tensor = csr_matrix(np.eye(m, n, dtype=int))

        # Save the sparse tensor
        self.save_sparse_tensor(sparse_tensor, "data_transformation")

        self.provenance_tensor = sparse_tensor
        return sparse_tensor

    @timed_capture
    def capture_vertical_augmentation(self, din, dout):
        n, m = len(din), len(dout)

        # Create a sparse identity matrix for vertical augmentation
        sparse_tensor = csr_matrix(np.eye(m, n, dtype=int))

        # Save the sparse tensor
        self.save_sparse_tensor(sparse_tensor, "vertical_augmentation")

        self.provenance_tensor = sparse_tensor
        return sparse_tensor

    def get_provenance(self, operation_key):
        return self.provenance_data.get(operation_key, None)

    def hash_row(self, row):
        """Generate a unique hash for a row."""
        return hash(tuple(row))

    def generate_hashed_df(self, df):
        """Generate a DataFrame with unique hashes for each row."""
        hashed_df = df.copy()
        hashed_df['hash'] = pd.util.hash_pandas_object(df, index=False)
        return hashed_df

    @timed_capture
    def capture_provenance_join(self, df_left, df_right, df_result, join_columns):
        # Step 1: Hash the input DataFrames
        df_left_hashed = self.generate_hashed_df(df_left)
        df_right_hashed = self.generate_hashed_df(df_right)
        df_result_hashed = self.generate_hashed_df(df_result)

        # Step 2: Project the result DataFrame onto the columns of the input DataFrames
        df_left_proj = df_result_hashed[df_left.columns].copy()
        df_right_proj = df_result_hashed[df_right.columns].copy()

        # Step 3: Generate hashes for the projections
        df_left_proj['hash_proj'] = df_left_proj['hash']
        df_right_proj['hash_proj'] = df_right_proj['hash']

        # Step 4: Recover provenance vectorized
        n_result = len(df_result)
        n_left = len(df_left)
        n_right = len(df_right)

        # Use a sparse tensor for join provenance
        sparse_tensor = csr_matrix((n_result, n_left * n_right), dtype=int)

        # Save the sparse tensor
        self.save_sparse_tensor(sparse_tensor, "join")

        self.provenance_data["join"] = sparse_tensor
        return sparse_tensor