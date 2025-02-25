import matplotlib.pyplot as plt

def plot_stock_price(df, ticker):
    """
    Plot the closing stock price over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    import data_fetcher
    df = data_fetcher.fetch_stock_data('AAPL', '2015-01-01', '2025-01-01')
    plot_stock_price(df, 'AAPL')
