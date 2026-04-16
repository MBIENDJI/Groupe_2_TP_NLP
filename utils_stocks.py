# utils_stocks.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from config import COMPANIES, WINDOW_SIZE

if not hasattr(np, 'NaN'):
    np.NaN = np.nan


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
    path       = f"lstm_{company.lower()}.pt"
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


# ============================================================
# DONNÉES
# ============================================================

def get_stock_data(company):
    ticker = COMPANIES[company]
    df = yf.download(
        ticker, period="5y",
        auto_adjust=True, progress=False)
    df.index = df.index.tz_localize(None)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ============================================================
# PRÉDICTIONS LSTM
# ============================================================

def predict_lstm(model, df, n_months):
    np.random.seed(42)
    scaler     = MinMaxScaler()
    scaler.fit(df[['Close']])
    n_days     = n_months * 21
    last_price = float(df['Close'].iloc[-1])
    volatility = float(df['Close'].pct_change().dropna().std())
    last_win   = df['Close'].values[-WINDOW_SIZE:].flatten()
    last_sc    = scaler.transform(
        last_win.reshape(-1, 1)).flatten()
    preds = []
    cur_w = last_sc.copy()
    with torch.no_grad():
        for _ in range(n_days):
            x  = torch.tensor(
                cur_w, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1)
            p  = model(x).numpy().flatten()[0]
            p += np.random.normal(0, volatility * 0.5)
            preds.append(p)
            cur_w = np.append(cur_w[1:], p)
    preds = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)).flatten()
    gap   = last_price - preds[0]
    fade  = np.linspace(1, 0, len(preds))
    preds = preds + gap * fade
    dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = n_days
    )
    return dates, preds


# ============================================================
# PROPHET — chargement JSON CORRIGÉ
# model_from_json attend une STRING, pas un dict
# ============================================================

def load_prophet_from_json(company):
    path = f"prophet_{company.lower()}.json"
    if not os.path.exists(path):
        return None
    try:
        from prophet.serialize import model_from_json
        with open(path, 'r') as f:
            json_str = f.read()          # lire comme STRING
        model = model_from_json(json_str)  # passer la STRING
        return model
    except Exception as e:
        print(f"Prophet JSON load error ({company}): {e}")
        return None


def train_prophet(df):
    from prophet import Prophet
    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y' : df['Close'].values.flatten()
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
    np.random.seed(42)
    n_days     = n_months * 21
    last_price = float(df['Close'].iloc[-1])
    volatility = float(df['Close'].pct_change().dropna().std())
    future     = pd.DataFrame({
        'ds': pd.bdate_range(
            start   = df.index[-1] + pd.Timedelta(days=1),
            periods = n_days
        )
    })
    forecast = model.predict(future)
    preds    = forecast['yhat'].values
    gap      = last_price - preds[0]
    fade     = np.linspace(1, 0, len(preds))
    preds    = preds + gap * fade
    noise    = np.random.normal(0, volatility, n_days)
    preds    = preds + noise * preds * np.linspace(
        0.3, 1.0, n_days)
    return future['ds'], preds


# ============================================================
# NEURALPROPHET — guard Python 3.14 incompatible
# ============================================================

NEURALPROPHET_AVAILABLE = None  # None = pas encore testé

def check_neuralprophet():
    """
    Vérifie si NeuralProphet est importable sur ce Python.
    Python 3.14 casse la lib holidays donc NeuralProphet crashe.
    """
    global NEURALPROPHET_AVAILABLE
    if NEURALPROPHET_AVAILABLE is not None:
        return NEURALPROPHET_AVAILABLE
    try:
        import neuralprophet  # noqa
        NEURALPROPHET_AVAILABLE = True
    except Exception:
        NEURALPROPHET_AVAILABLE = False
    return NEURALPROPHET_AVAILABLE


def train_neuralprophet(df):
    """Entraîne NeuralProphet — lève RuntimeError si indisponible."""
    if not check_neuralprophet():
        raise RuntimeError(
            "NeuralProphet incompatible avec Python "
            f"{sys.version_info.major}.{sys.version_info.minor} "
            "sur ce serveur. Utilisez LSTM et Prophet."
        )
    from neuralprophet import NeuralProphet
    neural_df = pd.DataFrame({
        'ds': df.index,
        'y' : df['Close'].values.flatten()
    }).reset_index(drop=True)
    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        n_lags            =14,
        epochs            =20,
        learning_rate     =0.001,
        batch_size        =32,
    )
    model.fit(neural_df, freq='B', progress='none')
    return model, neural_df


def predict_neuralprophet(model, train_df, df, n_months):
    np.random.seed(42)
    n_days     = n_months * 21
    last_price = float(df['Close'].iloc[-1])
    last_date  = pd.Timestamp(df.index[-1]).normalize()
    volatility = float(df['Close'].pct_change().dropna().std())
    try:
        future   = model.make_future_dataframe(
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
            raise ValueError("Pas de prédictions futures")
        preds = fut[col].values
        dates = fut['ds'].values
    except Exception as e:
        print(f"NeuralProphet predict fallback: {e}")
        trend = (float(df['Close'].iloc[-1]) -
                 float(df['Close'].iloc[-60])) / 60
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
    noise = np.random.normal(0, volatility, len(preds))
    preds = preds + noise * preds * np.linspace(
        0.3, 1.0, len(preds))
    return pd.to_datetime(dates), preds
