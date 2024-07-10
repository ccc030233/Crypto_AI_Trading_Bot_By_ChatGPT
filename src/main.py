import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import talib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess data from CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Return'] = data['close'].pct_change()
    data.dropna(inplace=True)

    # Add technical indicators manually
    data['SMA'] = talib.SMA(data['close'], timeperiod=14)
    data['EMA'] = talib.EMA(data['close'], timeperiod=14)
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['close'])
    data['Upper_BB'], data['Middle_BB'], data['Lower_BB'] = talib.BBANDS(data['close'])

    # Add lag features
    for lag in range(1, 11):
        data[f'lag_{lag}'] = data['Return'].shift(lag)
    
    data.dropna(inplace=True)
    
    # Drop non-numeric columns
    data = data.select_dtypes(include=[np.number])
    return data

# Create dataset
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back)]
        X.append(a)
        Y.append(data[i + look_back, 0])  # Assume 'Return' is the first column
    return np.array(X), np.array(Y)

data = load_data('data/my_dataframe_1000.csv')
target_column = 'Return'
features = data.drop(columns=[target_column]).values
targets = data[target_column].values

# Normalize the dataset
scaler = StandardScaler()
features = scaler.fit_transform(features)

look_back = 10
X, Y = create_dataset(features, look_back)

# Debugging statements to check the shapes
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Debugging statements to check the tensor shapes
print(f"Shape of X_tensor: {X_tensor.shape}")
print(f"Shape of Y_tensor: {Y_tensor.shape}")

# Create DataLoader with train and validation split
dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Experiment with different batch sizes
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define an LSTM network
class LSTMNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

input_size = X.shape[2]  # Number of features
hidden_size = 100
num_layers = 2
output_size = 1
model = LSTMNN(input_size, hidden_size, num_layers, output_size)

# Experiment with different learning rates
learning_rate = 0.0005

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop with early stopping and validation
num_epochs = 50
patience = 5
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        
        # Ensure targets shape matches outputs shape
        targets = targets.view_as(outputs)
        
        loss = loss_fn(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            targets = targets.view_as(outputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    scheduler.step()
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logging.info('Early stopping!')
            break

print("Training complete.")