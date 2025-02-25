import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd

def run_arima_forecast(df, ticker):
    """
    Build and evaluate an ARIMA model for forecasting the closing price.
    """
    # Use 'Close' price for forecasting
    ts = df['Close']

    # Split into training and test sets (80% train, 20% test)
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    # Fit ARIMA model (parameters p=5, d=1, q=0 - can be tuned)
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast for the length of the test set
    forecast = model_fit.forecast(steps=len(test))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f'ARIMA Model RMSE: {rmse:.2f}')

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Test Data')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title(f'{ticker} Stock Price Forecast using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    from data_fetcher import fetch_stock_data
    df = fetch_stock_data('AAPL', '2015-01-01', '2023-01-01')
    run_arima_forecast(df, 'AAPL')
