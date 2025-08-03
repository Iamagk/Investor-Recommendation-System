import pandas as pd
import numpy as np

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- SMA ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # --- EMA ---
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # --- RSI ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    # --- Bollinger Bands ---
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()

    # --- ATR ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    # --- ADX ---
    df['TR'] = true_range
    df['+DM'] = np.where((df['High'].diff() > df['Low'].diff()) & (df['High'].diff() > 0), df['High'].diff(), 0)
    df['-DM'] = np.where((df['Low'].diff() > df['High'].diff()) & (df['Low'].diff() > 0), df['Low'].diff(), 0)
    tr_14 = df['TR'].rolling(window=14).sum()
    plus_di_14 = 100 * df['+DM'].rolling(window=14).sum() / tr_14
    minus_di_14 = 100 * df['-DM'].rolling(window=14).sum() / tr_14
    dx = (np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100
    df['ADX_14'] = dx.rolling(window=14).mean()

    # --- Stochastic RSI ---
    min_rsi = df['RSI_14'].rolling(window=14).min()
    max_rsi = df['RSI_14'].rolling(window=14).max()
    df['StochRSI'] = (df['RSI_14'] - min_rsi) / (max_rsi - min_rsi)

    # --- On-Balance Volume (OBV) ---
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    return df