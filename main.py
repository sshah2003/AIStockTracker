from src import data_fetcher, eda, model

def main():
    # Parameters
    ticker = 'AAPL'
    start_date = '2024-12-01'
    end_date = '2025-01-31'
    
    # Step 1: Fetch data
    df = data_fetcher.fetch_stock_data(ticker, start_date, end_date)
    
    # Step 2: Perform EDA
    eda.plot_stock_price(df, ticker)
    
    # Step 3: Build and evaluate the ARIMA model
    model.run_auto_arima_forecast(df, ticker)

if __name__ == '__main__':
    main()
