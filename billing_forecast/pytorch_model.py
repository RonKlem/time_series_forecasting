import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


"""
pytorch_model.py
----------------
Contains a small PyTorch-based model and training utilities for forecasting.

Design notes for non-technical readers:
- The model is a simple LSTM (a kind of recurrent neural network) that can
    use numeric features and small learned representations (embeddings) of
    categorical variables like customer and product codes.
- We keep the dataset format simple: each training row contains numeric
    features followed by three integers that identify the customer, product
    area, and product code. The training loop extracts these parts before
    feeding them to the model.
- The training function includes early stopping: it stops training when the
    validation error doesn't improve for `patience` epochs. This avoids
    overfitting and keeps training time reasonable.

This file also includes an Optuna-based tuner that runs a limited search
over model hyperparameters (sizes, learning rate, etc.). Optuna is optional
and the tuning is skipped if it is not installed.
"""


class TimeSeriesDataset(Dataset):
    """
    Lightweight dataset wrapper for the training loop.

    The dataset stores X (numpy array) and y (numpy array). For simplicity we
    treat each row as a single sample. If you want true sequences (many
    historical timesteps per sample), this class can be extended to return
    sequences instead.
    """

    def __init__(self, X, y, seq_len=12):
        # Convert to float32 which PyTorch expects
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single row (features) and its target value
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, num_numeric, cust_emb_size, num_customers, pa_emb_size, num_pa, pc_emb_size, num_pc, hidden_size=64, n_layers=1):
        super().__init__()
        """
        LSTM model with small embedding layers for categorical variables.

        Args:
            num_numeric: number of numeric features per row
            cust_emb_size: size of the customer embedding vector
            num_customers: number of unique customers
            pa_emb_size, num_pa: product area embedding size and cardinality
            pc_emb_size, num_pc: product code embedding size and cardinality
            hidden_size: LSTM hidden state size
            n_layers: number of stacked LSTM layers
        """
        # Embeddings: these learn a small vector for each unique category.
        self.cust_emb = nn.Embedding(num_customers, cust_emb_size)
        self.pa_emb = nn.Embedding(num_pa, pa_emb_size)
        self.pc_emb = nn.Embedding(num_pc, pc_emb_size)

        # The LSTM input is the numeric features concatenated with the embeddings
        input_size = num_numeric + cust_emb_size + pa_emb_size + pc_emb_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=n_layers)

        # Simple feed-forward head to produce a single scalar prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_numeric, cust_idx, pa_idx, pc_idx):
        # x_numeric can be (B, num_numeric) for single-step or
        # (B, T, num_numeric) if you adapt the dataset for sequences.
        if x_numeric.dim() == 2:
            x_numeric = x_numeric.unsqueeze(1)  # make it (B, 1, num_numeric)
        B, T, _ = x_numeric.shape

        # Get embedding vectors and repeat them across the time dimension
        cust_e = self.cust_emb(cust_idx).unsqueeze(1).repeat(1, T, 1)
        pa_e = self.pa_emb(pa_idx).unsqueeze(1).repeat(1, T, 1)
        pc_e = self.pc_emb(pc_idx).unsqueeze(1).repeat(1, T, 1)

        # Concatenate numeric features with embeddings and feed into LSTM
        x = torch.cat([x_numeric, cust_e, pa_e, pc_e], dim=-1)
        out, _ = self.lstm(x)
        # Use the last output of the LSTM (many-to-one) then pass through head
        out = out[:, -1, :]
        out = self.fc(out).squeeze(-1)
        return out


def train_torch_model(train_dataset, val_dataset, model, epochs=50, batch_size=64, lr=1e-3, patience=5, device=None):
    """
    Train a PyTorch model with early stopping.

    Args:
        train_dataset, val_dataset: dataset objects that yield (X_row, y)
        model: a PyTorch nn.Module
        epochs: maximum training epochs
        batch_size: training batch size
        lr: learning rate for the optimizer
        patience: number of epochs with no improvement on validation loss
                  before training stops (early stopping).
        device: 'cpu' or 'cuda' (GPU). If None, the function chooses GPU when
                available.

    Returns:
        The model with weights restored to the best validation epoch.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # mean squared error loss for regression

    # Data loaders wrap the datasets and provide batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        # Training loop over batches
        for X_batch, y_batch in train_loader:
            # Each X_batch is expected to be a numeric array with the
            # categorical indices appended as the last 3 columns:
            # [...numeric features..., customer_idx, product_area_idx, product_code_idx]
            X_batch = torch.tensor(X_batch).to(device)
            y_batch = y_batch.to(device)

            # Extract categorical indices from the last columns
            cust_idx = X_batch[:, -3].long()
            pa_idx = X_batch[:, -2].long()
            pc_idx = X_batch[:, -1].long()
            numeric = X_batch[:, :-3]

            preds = model(numeric, cust_idx, pa_idx, pc_idx)
            loss = loss_fn(preds, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_dataset)

        # Validation loop (no gradient updates)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = torch.tensor(X_batch).to(device)
                y_batch = y_batch.to(device)
                cust_idx = X_batch[:, -3].long()
                pa_idx = X_batch[:, -2].long()
                pc_idx = X_batch[:, -1].long()
                numeric = X_batch[:, :-3]
                preds = model(numeric, cust_idx, pa_idx, pc_idx)
                loss = loss_fn(preds, y_batch)
                val_loss += loss.item() * len(y_batch)
        val_loss /= len(val_dataset)

        # Early stopping logic: if validation loss improves we save the model
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def optuna_tune_pytorch(X_train, y_train, X_val, y_val, num_numeric, num_customers, num_pa, num_pc, timeout=300):
    """
    Optionally tune PyTorch model hyperparameters using Optuna.

    This function runs an automated search over model sizes and learning rates
    and returns an Optuna `Study` object containing the best parameters.

    If Optuna is not installed the function prints a message and returns None.
    """
    try:
        import optuna
    except Exception:
        print('Optuna not installed; skipping PyTorch hyperparameter tuning')
        return None

    def objective(trial):
        # Suggest hyperparameters to try. Optuna will pick combinations to
        # minimize the objective (validation RMSE).
        hidden_size = trial.suggest_int('hidden_size', 16, 256, log=True)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        # dropout is not currently used in the model, but we keep it here to
        # show how additional parameters could be added.
        dropout = trial.suggest_float('dropout', 0.0, 0.5)

        model = LSTMModel(
            num_numeric=num_numeric,
            cust_emb_size=trial.suggest_int('cust_emb', 4, 32),
            num_customers=num_customers,
            pa_emb_size=trial.suggest_int('pa_emb', 2, 16),
            num_pa=num_pa,
            pc_emb_size=trial.suggest_int('pc_emb', 4, 24),
            num_pc=num_pc,
            hidden_size=hidden_size,
            n_layers=n_layers
        )

        # Prepare small datasets for the trial
        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds = TimeSeriesDataset(X_val, y_val)

        # Train briefly and evaluate on the validation split. Trials are
        # intentionally short to keep tuning time reasonable.
        model = train_torch_model(train_ds, val_ds, model, epochs=30, batch_size=128, lr=lr, patience=5)

        # Evaluate RMSE on validation set
        loader = DataLoader(val_ds, batch_size=128)
        preds = []
        ys = []
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        with torch.no_grad():
            for Xb, yb in loader:
                Xb = torch.tensor(Xb).to(device)
                cust_idx = Xb[:, -3].long()
                pa_idx = Xb[:, -2].long()
                pc_idx = Xb[:, -1].long()
                numeric = Xb[:, :-3]
                p = model(numeric, cust_idx, pa_idx, pc_idx).cpu().numpy()
                preds.append(p)
                ys.append(yb.numpy())
        preds = np.concatenate(preds)
        ys = np.concatenate(ys)
        rmse_val = math.sqrt(((preds - ys) ** 2).mean())
        return rmse_val

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, timeout=timeout)
    return study
