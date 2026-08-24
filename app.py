from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    stock = request.form['ticker'].upper().strip()

    # ================= DOWNLOAD DATA =================
    df = yf.download(stock, period='1y', auto_adjust=False)

    if df.empty:
        return render_template('index.html', error="Invalid Stock Ticker")

    # ================= CLEAN DATA =================
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df.index = pd.to_datetime(df.index)

    # ================= MOVING AVERAGES =================
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    df['Signal'] = 0
    df.loc[df['MA20'] > df['MA50'], 'Signal'] = 1
    df.loc[df['MA20'] < df['MA50'], 'Signal'] = -1

    latest_signal = df['Signal'].iloc[-1]

    if latest_signal == 1:
        signal_text = "📈 BUY"
    elif latest_signal == -1:
        signal_text = "📉 SELL"
    else:
        signal_text = "⚖ HOLD"

    # ================= ML MODEL =================
    df['Prev_Close'] = df['Close'].shift(1)
    df = df.dropna()

    X = df[['Prev_Close']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    # Predictions for evaluation
    y_pred = model.predict(X)

    # ================= ERROR METRICS =================
    mae = round(mean_absolute_error(y, y_pred), 2)
    rmse = round(np.sqrt(mean_squared_error(y, y_pred)), 2)

    # ================= NEXT DAY PREDICTION =================
    last_close = df['Close'].iloc[-1]
    predicted_price = model.predict([[last_close]])[0]
    predicted_price = round(float(predicted_price), 2)

    current_price = round(float(last_close), 2)

    # ================= LINE CHART =================
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label='Close Price')
    plt.title(f"{stock} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    chart_path = f"static/{stock}_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # ================= CANDLESTICK CHART =================
    candlestick_path = f"static/{stock}_candlestick.png"

    mpf.plot(
        df,
        type='candle',
        style='yahoo',
        title=f'{stock} Candlestick Chart',
        volume=True,
        savefig=candlestick_path
    )

    # ================= RENDER =================
    return render_template(
        'index.html',
        company_name=stock,
        sector="Stock Market",
        market_cap="Live Data",
        current_price=current_price,
        predicted_price=predicted_price,
        signal=signal_text,
        mae=mae,
        rmse=rmse,
        plot1=chart_path,
        plot2=candlestick_path
    )


if __name__ == '__main__':
    app.run(port=5001, debug=True)
