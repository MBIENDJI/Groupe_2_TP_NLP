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


# ============================================================
# FONCTION UTILITAIRE — Aplatir les colonnes yfinance
# ============================================================

def flatten_df(df):
    """
    Corrige le MultiIndex de yfinance (Python 3.14).
    Retourne un DataFrame propre avec colonnes simples.
    """
    if df is None or df.empty:
        return df

    # Aplatir MultiIndex si présent
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Forcer chaque colonne en Series 1D
    for col in df.columns:
        if isinstance(df[col].iloc[0], pd.Series):
            df[col] = df[col].apply(
                lambda x: float(x.iloc[0])
                if isinstance(x, pd.Series) else float(x)
            )

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
    """Charge le modèle LSTM."""
    path = f"lstm_{company.lower()}.pt"
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and \
       'model_state' in checkpoint:
        model = StockLSTM(
            hidden_size = checkpoint.get('hidden_size', 64),
            num_layers  = checkpoint.get('num_layers', 2),
            dropout     = checkpoint.get('dropout', 0.2)
        )
        model.load_state_dict(checkpoint['model_state'])
    else:
        model = StockLSTM()
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# ============================================================
# DONNÉES
# ============================================================

def get_stock_data(company):
    """Télécharge et nettoie les données Yahoo Finance."""
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

        # S'assurer que Close est bien numérique 1D
        df['Close'] = pd.to_numeric(
            df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])

        return df

    except Exception as e:
        print(f"Erreur {ticker}: {e}")
        return None


def get_close_value(df):
    """
    Extrait le dernier prix de clôture en float.
    Gère tous les cas MultiIndex/Series imbriquées.
    """
    val = df['Close'].iloc[-1]

    # Cas Series imbriquée
    if isinstance(val, pd.Series):
        val = val.iloc[0]

    # Cas numpy array
    if hasattr(val, '__len__'):
        val = float(val.flat[0])

    return float(val)


# ============================================================
# PRÉDICTIONS LSTM
# ============================================================

def predict_lstm(model, df, n_months):
    """Prédit les prix futurs avec LSTM."""
    np.random.seed(42)

    close_vals = df['Close'].values.flatten().astype(float)
    last_price = float(close_vals[-1])
    volatility = float(
        pd.Series(close_vals).pct_change().dropna().std())

    scaler = MinMaxScaler()
    scaler.fit(close_vals.reshape(-1, 1))

    n_days = n_months * 21

    if len(close_vals) < WINDOW_SIZE:
        dates = pd.bdate_range(
            start   = df.index[-1] + pd.Timedelta(days=1),
            periods = n_days
        )
        return dates, np.array([last_price] * n_days)

    last_win = close_vals[-WINDOW_SIZE:]
    last_sc  = scaler.transform(
        last_win.reshape(-1, 1)).flatten()

    preds = []
    cur_w = last_sc.copy()

    with torch.no_grad():
        for _ in range(n_days):
            x = torch.tensor(
                cur_w, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1)
            p  = model(x).numpy().flatten()[0]
            p += np.random.normal(0, volatility * 0.5)
            preds.append(p)
            cur_w = np.append(cur_w[1:], p)

    preds = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

    gap   = last_price - preds[0]
    fade  = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade

    dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = n_days
    )
    return dates, preds


# ============================================================
# PROPHET
# ============================================================

def load_prophet_from_json(company):
    """Charge Prophet depuis JSON."""
    path = f"prophet_{company.lower()}.json"
    if not os.path.exists(path):
        return None
    try:
        from prophet.serialize import model_from_json
        with open(path, 'r') as f:
            return model_from_json(f.read())
    except Exception as e:
        print(f"Prophet JSON error: {e}")
        return None


def train_prophet(df):
    """Entraîne Prophet à la volée."""
    from prophet import Prophet

    close_vals = df['Close'].values.flatten().astype(float)

    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y' : close_vals
    }).reset_index(drop=True)

    model = Prophet(
        yearly_seasonality      = True,
        weekly_seasonality      = True,
        daily_seasonality       = False,
        changepoint_prior_scale = 0.05
    )
    model.fit(prophet_df)
    return model, prophet_df


def predict_prophet(model, df, n_months):
    """Prédit les prix futurs avec Prophet."""
    np.random.seed(42)

    close_vals = df['Close'].values.flatten().astype(float)
    last_price = float(close_vals[-1])
    n_days     = n_months * 21

    future = pd.DataFrame({
        'ds': pd.bdate_range(
            start   = df.index[-1] + pd.Timedelta(days=1),
            periods = n_days
        )
    })

    forecast = model.predict(future)
    preds    = forecast['yhat'].values

    gap   = last_price - preds[0]
    fade  = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade

    return future['ds'], preds


# ============================================================
# NEURALPROPHET
# ============================================================

NEURALPROPHET_AVAILABLE = None


def check_neuralprophet():
    global NEURALPROPHET_AVAILABLE
    if NEURALPROPHET_AVAILABLE is not None:
        return NEURALPROPHET_AVAILABLE
    try:
        import neuralprophet
        NEURALPROPHET_AVAILABLE = True
    except Exception:
        NEURALPROPHET_AVAILABLE = False
    return NEURALPROPHET_AVAILABLE


def train_neuralprophet(df):
    """Entraîne NeuralProphet à la volée."""
    from neuralprophet import NeuralProphet

    close_vals = df['Close'].values.flatten().astype(float)

    neural_df = pd.DataFrame({
        'ds': df.index,
        'y' : close_vals
    }).reset_index(drop=True)

    model = NeuralProphet(
        yearly_seasonality = True,
        weekly_seasonality = True,
        daily_seasonality  = False,
        n_lags             = 14,
        epochs             = 20,
        learning_rate      = 0.001,
        batch_size         = 32,
    )
    model.fit(neural_df, freq='B', progress='none')
    return model, neural_df


def predict_neuralprophet(model, train_df, df, n_months):
    """Prédit les prix futurs avec NeuralProphet."""
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan

    np.random.seed(42)
    close_vals = df['Close'].values.flatten().astype(float)
    last_price = float(close_vals[-1])
    last_date  = pd.Timestamp(df.index[-1]).normalize()
    n_days     = n_months * 21

    try:
        future = model.make_future_dataframe(
            df                     = train_df,
            periods                = n_days,
            n_historic_predictions = True
        )
        forecast = model.predict(future)
        col      = ('yhat1' if 'yhat1' in forecast.columns
                    else 'yhat')
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        fut = forecast[
            forecast['ds'] > last_date
        ].dropna(subset=[col]).reset_index(drop=True)

        if len(fut) == 0:
            raise ValueError("Pas de prédictions")

        preds = fut[col].values.astype(float)
        dates = fut['ds'].values

    except Exception as e:
        print(f"NeuralProphet fallback: {e}")
        trend = (close_vals[-1] - close_vals[-60]) / 60
        preds = np.array([
            last_price + trend * i
            for i in range(1, n_days + 1)
        ])
        dates = pd.bdate_range(
            start   = df.index[-1] + pd.Timedelta(days=1),
            periods = n_days
        )

    gap   = last_price - preds[0]
    fade  = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade

    return pd.to_datetime(dates), preds
