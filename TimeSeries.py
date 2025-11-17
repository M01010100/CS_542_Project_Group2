from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataHelper import load_data

def preprocess_dates():
    data = load_data('data/Groceries_dataset.csv')
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date')    
    df = df.dropna(subset=['Date'])
    
    return df

def aggregate_daily_data(data, item=None):
    #- item: Specific item to analyze (None for all items)    
    if item:
        data = data[data['itemDescription'] == item]
    daily_data = data.groupby('Date').size().reset_index(name='quantity')
    
    return daily_data

def time_series_forecast(item=None, forecast_days=30):
    #Parameters:
    #- item: Specific item to forecast (None for overall sales)
    #- forecast_days: Number of days to forecast
    print("=" * 80)
    print("TIME SERIES FORECASTING")
    print("=" * 80)
    
    print("\n Preprocessing dates...")
    df = preprocess_dates()
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total days: {(df['Date'].max() - df['Date'].min()).days}")
    
    print("\n Aggregating data...")
    if item:
        print(f"Analyzing item: {item}")
    else:
        print("Analyzing overall transaction volume")
    
    daily_data = aggregate_daily_data(df, item)
    print(f"Total data points: {len(daily_data)}")
    
    prophet_df = daily_data.rename(columns={'Date': 'ds', 'quantity': 'y'})
    
    date_range = pd.date_range(start=prophet_df['ds'].min(), 
                                end=prophet_df['ds'].max(), 
                                freq='D')
    prophet_df = prophet_df.set_index('ds').reindex(date_range, fill_value=0).reset_index()
    prophet_df.columns = ['ds', 'y']
    
    print("\n Training forecasting model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_df)
    
    # Step 5: Make predictions
    print(f"\n Forecasting next {forecast_days} days...")
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    print("\n" + "=" * 80)
    print("FORECAST SUMMARY")
    print("=" * 80)
    
    historical_end = len(prophet_df)

    print("\n Last 5 Historical Days:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[historical_end-5:historical_end].to_string(index=False))

    print("\n Next 5 Forecasted Days:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[historical_end:historical_end+5].to_string(index=False))
    
    # Step 7: Calculate metrics
    mae = np.mean(np.abs(prophet_df['y'] - forecast['yhat'][:len(prophet_df)]))
    print(f"\n Mean Absolute Error: {mae:.2f}")
    
    # Step 8: Visualizations
    print("\n Generating visualizations...")

    # Plot forecast
    fig1 = model.plot(forecast)
    plt.title(f'Sales Forecast - {item if item else "Overall"}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.tight_layout()
    
    # Plot components (trend, seasonality)
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    
    plt.show()
    
    print("=" * 80 + "\n")
    
    return forecast, model

def analyze_multiple_items():
    data = preprocess_dates()
    #Forecast for top N most popular items
    forecast_days = input("Enter number of days to forecast (default 30): ")
    forecast_days = int(forecast_days) if forecast_days else 30
    top_n = input("Enter number of top items to forecast (default 5): ")
    top_n = int(top_n) if top_n else 5

    top_items = data['itemDescription'].value_counts().head(top_n)
    
    print(f"\n📦 Forecasting for top {top_n} items:")
    print(top_items)
    
    forecasts = {}
    for item in top_items.index:
        print(f"\n{'='*80}")
        print(f"Analyzing: {item}")
        print('='*80)
        forecast, model = time_series_forecast(data, item=item, forecast_days=forecast_days)
        forecasts[item] = forecast
    
    return forecasts