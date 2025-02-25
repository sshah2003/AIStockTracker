import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.model import fetch_stock_data, run_auto_arima_forecast
from pmdarima import auto_arima
import numpy as np
from sklearn.metrics import mean_squared_error

st.title("Interactive Stock Forecast Dashboard")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Button to trigger data fetching and forecasting
if st.sidebar.button("Run Forecast"):
    df = fetch_stock_data(ticker, str(start_date), str(end_date))
    
    # Split data into training and testing sets (80% train, 20% test)
    ts = df['Close']
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # Fit auto_arima model on training data
    model = auto_arima(train,
                       seasonal=True,
                       m=5,
                       trace=False,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)
    forecast = model.predict(n_periods=len(test))
    rmse = np.sqrt(mean_squared_error(test, forecast))
    
    st.write("## Model Summary")
    st.text(model.summary())
    st.write(f"### Refined ARIMA Model RMSE: {rmse:.2f}")
    
    # Plot interactive graph with Plotly
    st.write("### Data Preview", df.head())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Data'))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test Data'))
    fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
    fig.update_layout(title=f'{ticker} Stock Price Forecast',
                      xaxis_title='Date',
                      yaxis_title='Price ($)',
                      hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display raw data
    st.write("## Raw Stock Data")
    st.dataframe(df.tail(10))
