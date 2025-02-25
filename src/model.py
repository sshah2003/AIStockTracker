import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.index = pd.to_datetime(df.index)
    return df.ffill()

def run_auto_arima_forecast(df, ticker, exog=None):
    """
    Refine the ARIMA model using auto_arima.
    Optionally, pass exogenous data as 'exog' (must be aligned with the time series).
    """
    ts = df['Close']
    
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # If exogenous data is provided, split it as well
    exog_train, exog_test = None, None
    if exog is not None:
        exog_train, exog_test = exog[:train_size], exog[train_size:]
    
    # Run auto_arima to determine the best (p,d,q) and seasonal (P,D,Q,m) parameters.
    # For seasonal data, set seasonal=True and m to the number of periods per season (e.g., m=5 for business days per week).
    model = auto_arima(train,
                       exogenous=exog_train,
                       seasonal=True,  # try seasonal model if you suspect seasonality
                       m=5,  # assuming business days; adjust as needed
                       trace=True,
                       error_action='ignore',  
                       suppress_warnings=True,
                       stepwise=True)
    
    print("Best model:", model.summary())
    
    # Forecast the test set
    forecast = model.predict(n_periods=len(test), exogenous=exog_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f'Refined ARIMA Model RMSE: {rmse:.2f}')
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Test Data')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title(f'{ticker} Stock Price Forecast (Refined Model)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    ticker = 'AAPL'
    df = fetch_stock_data(ticker, '2015-01-01', '2023-01-01')
    
    # Example: if you had exogenous variables, you would prepare them here.
    # For now, we run without exogenous data.
    run_auto_arima_forecast(df, ticker)
