import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# --- 1. Configuration and File Paths ---
# Adjust BASE_DIR to where your 'SMAP' folder is located
BASE_DIR = './data/SMAP/' 

TRAIN_DATA_PATH = BASE_DIR + 'SMAP_train.npy'
TEST_DATA_PATH = BASE_DIR + 'SMAP_test.npy'
TEST_LABEL_PATH = BASE_DIR + 'SMAP_test_label.npy'

# GDN Hyperparameters
WINDOW_SIZE = 100  # Look-back window length (hyperparameter 'k' in the paper)
BATCH_SIZE = 64

# --- 2. Data Loading and Shape Inspection ---
print("--- Loading Raw Data ---")
X_train_raw = np.load(TRAIN_DATA_PATH)
X_test_raw = np.load(TEST_DATA_PATH)
y_test_raw = np.load(TEST_LABEL_PATH)

print(f"X_train_raw shape: {X_train_raw.shape}")
print(f"X_test_raw shape: {X_test_raw.shape}")
print(f"y_test_raw shape: {y_test_raw.shape}")

# Expected shapes (Time steps, Features/Dimensions):
# T_train x D (approx. 138k x 25)
# T_test x D (approx. 435k x 25)
# T_test x 1 (binary labels)
T_train, D_features = X_train_raw.shape

# --- 3. Normalization (Scaling) ---
# GDN is a neural network model, so data must be scaled.
# Crucially, the scaler must be fitted ONLY on the training data to prevent leakage.
print("\n--- Applying Normalization (MinMax Scaling) ---")
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit only on training data
scaler.fit(X_train_raw)

# Transform both training and testing data
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Convert back to NumPy for the sliding window step
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)
y_test_raw = y_test_raw.astype(np.float32)

# --- 4. Sliding Window Transformation ---

def create_sliding_windows(data, window_size):
    """
    Creates overlapping sequences (sliding windows) from the time series data.
    Input: (T, D) -> Output: (N_samples, Window_Size, D)
    """
    N_time, N_dims = data.shape
    windows = []
    # Loop from 0 up to the point where a full window can be created
    for i in range(N_time - window_size):
        windows.append(data[i : i + window_size, :])
    return np.array(windows)

# Create windows for training data (X_train) and testing data (X_test)
X_train_windows = create_sliding_windows(X_train_scaled, WINDOW_SIZE)
X_test_windows = create_sliding_windows(X_test_scaled, WINDOW_SIZE)

# For anomaly detection, the GDN model needs to predict the *next* time step (t+1).
# The target y is the time step *after* the window: data[i + window_size, :]
# The anomaly detection target y_test_label is for the predicted point (t+1).
y_train_windows = X_train_scaled[WINDOW_SIZE:, :]  # Target is the next point
y_test_windows = X_test_scaled[WINDOW_SIZE:, :]
y_test_labels_windows = y_test_raw[WINDOW_SIZE:] # Anomaly label for the predicted point

print(f"\nX_train_windows shape: {X_train_windows.shape}") 
print(f"y_train_windows shape: {y_train_windows.shape}")
print(f"X_test_windows shape: {X_test_windows.shape}")
print(f"y_test_windows shape: {y_test_windows.shape}")
print(f"y_test_labels_windows shape: {y_test_labels_windows.shape}")

# --- 5. PyTorch DataLoader Creation ---
X_train_tensor = torch.from_numpy(X_train_windows)
y_train_tensor = torch.from_numpy(y_train_windows)

X_test_tensor = torch.from_numpy(X_test_windows)
y_test_tensor = torch.from_numpy(y_test_windows)
y_test_labels_tensor = torch.from_numpy(y_test_labels_windows)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, y_test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\n--- DataLoader Creation Complete ---")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")
print("Data is ready for the GDN model training.")
