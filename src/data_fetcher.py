import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for the given ticker and date range.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    # Ensure datetime index and handle missing values
    df.index = pd.to_datetime(df.index)
    df = df.ffill()  # Forward fill missing data
    return df

if __name__ == '__main__':
    # For quick testing
    data = fetch_stock_data('AAPL', '2015-01-01', '2023-01-01')
    print(data.head())
