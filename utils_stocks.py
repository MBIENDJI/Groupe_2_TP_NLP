# utils_stocks.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from config import COMPANIES, WINDOW_SIZE

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Chemin racine du projet (là où sont les .pt et .json)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# FONCTION UTILITAIRE — Aplatir les colonnes yfinance
# ============================================================

def flatten_df(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    for col in df.columns:
        try:
            if isinstance(df[col].iloc[0], pd.Series):
                df[col] = df[col].apply(
                    lambda x: float(x.iloc[0])
                    if isinstance(x, pd.Series) else float(x)
                )
        except Exception:
            pass
    return df


# ============================================================
# LSTM
# ============================================================

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers,
                         x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers,
                         x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def load_lstm(company):
    # Chercher à la racine du projet
    path = os.path.join(BASE_DIR, f"lstm_{company.lower()}.pt")

    # Fallback : même dossier que ce fichier
    if not os.path.exists(path):
        path = f"lstm_{company.lower()}.pt"

    if not os.path.exists(path):
        print(f"LSTM non trouvé : {path}")
        return None

    try:
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and \
           'model_state' in checkpoint:
            model = StockLSTM(
                hidden_size=checkpoint.get('hidden_size', 64),
                num_layers =checkpoint.get('num_layers', 2),
                dropout    =checkpoint.get('dropout', 0.2)
            )
            model.load_state_dict(checkpoint['model_state'])
        else:
            model = StockLSTM()
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        print(f"Erreur chargement LSTM {company}: {e}")
        return None


# ============================================================
# DONNÉES
# ============================================================

def get_stock_data(company):
    ticker = COMPANIES[company]
    try:
        df = yf.download(
            ticker, period="2y",
            auto_adjust=True, progress=False
        )
        if df is None or df.empty:
            return None
        df.index = df.index.tz_localize(None)
        df = flatten_df(df)
        if 'Close' not in df.columns:
            return None
        df['Close'] = pd.to_numeric(
            df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        return df
    except Exception as e:
        print(f"Erreur {ticker}: {e}")
        return None


def get_close_value(df):
    val = df['Close'].values.flatten().astype(float)[-1]
    return float(val)


# ============================================================
# PRÉDICTIONS LSTM
# ============================================================

def predict_lstm(model, df, n_months):
    np.random.seed(42)
    close_vals = df['Close'].values.flatten().astype(float)
    last_price = float
