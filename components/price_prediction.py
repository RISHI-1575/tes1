import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os

# Import the price prediction model
from models.price_predictor import CropPricePredictor

def render_price_prediction():
    st.header("Crop Price Prediction")
    st.write("Predict crop prices for the next 6 months based on historical data.")
    
    # Load datasets
    try:
        df = pd.read_csv('data/crop_prices.csv')
        
        # Get unique cities and crops
        cities = sorted(df['city'].unique())
        crops = sorted(df['crop_name'].unique())
        
        # User input
        col1, col2 = st.columns(2)
        
        with col1:
            selected_city = st.selectbox("Select City", cities)
        
        with col2:
            selected_crop = st.selectbox("Select Crop", crops)
        
        # Filter data for selected city and crop
        filtered_data = df[(df['city'] == selected_city) & (df['crop_name'] == selected_crop)]
        
        if filtered_data.empty:
            st.warning(f"No data available for {selected_crop} in {selected_city}")
            return
        
        # Convert date to datetime
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        
        # Sort by date
        filtered_data = filtered_data.sort_values('date')
        
        # Display historical data
        st.subheader("Historical Price Data")
        st.dataframe(filtered_data[['date', 'price', 'quantity_sold']].set_index('date'))
        
        # Create price predictor instance
        predictor = CropPricePredictor()
        
        # Make prediction
        try:
            # Check if model file exists
            model_path = f'models/saved/{selected_city}_{selected_crop}_price_model.pkl'
            
            if os.path.exists(model_path):
                # Load existing model
                with open(model_path, 'rb') as f:
                    predictor.model = pickle.load(f)
            else:
                # Train new model
                st.info("Training model for the first time... This may take a moment.")
                predictor.train(filtered_data)
                
                # Create directory if it doesn't exist
                os.makedirs('models/saved', exist_ok=True)
                
                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump(predictor.model, f)
            
            # Get the last date in the dataset
            last_date = filtered_data['date'].max()
            
            # Generate next 6 months dates
            future_dates = [last_date + timedelta(days=30*i) for i in range(1, 7)]
            future_dates_str = [date.strftime('%Y-%m') for date in future_dates]
            
            # Make prediction for the next 6 months
            predicted_prices = predictor.predict(filtered_data, n_months=6)
            
            # Create a dataframe with future dates and predicted prices
            future_df = pd.DataFrame({
                'date': future_dates_str,
                'predicted_price': predicted_prices
            })
            
            # Display predictions
            st.subheader("Price Predictions for Next 6 Months")
            st.dataframe(future_df)
            
            # Plot historical and predicted prices
            st.subheader("Price Trend and Prediction")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical prices
            historical_dates = filtered_data['date'].dt.strftime('%Y-%m').tolist()
            historical_prices = filtered_data['price'].tolist()
            ax.plot(historical_dates, historical_prices, marker='o', linestyle='-', label='Historical Prices')
            
            # Plot predicted prices
            ax.plot(future_dates_str, predicted_prices, marker='s', linestyle='--', color='red', label='Predicted Prices')
            
            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹/quintal)')
            ax.set_title(f'{selected_crop} Price Trend in {selected_city}')
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Show plot
            st.pyplot(fig)
            
            # Additional insights
            avg_price = filtered_data['price'].mean()
            max_price = filtered_data['price'].max()
            min_price = filtered_data['price'].min()
            
            max_predicted = max(predicted_prices)
            min_predicted = min(predicted_prices)
            
            st.subheader("Price Insights")
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Historical Price", f"₹{avg_price:.2f}")
            col2.metric("Highest Historical Price", f"₹{max_price:.2f}")
            col3.metric("Lowest Historical Price", f"₹{min_price:.2f}")
            
            col1, col2 = st.columns(2)
            col1.metric("Highest Predicted Price", f"₹{max_predicted:.2f}")
            col2.metric("Lowest Predicted Price", f"₹{min_predicted:.2f}")
            
            # Price seasonality
            st.subheader("Price Seasonality")
            filtered_data['month'] = filtered_data['date'].dt.month_name()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_avg = filtered_data.groupby('month')['price'].mean().reindex(month_order)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=monthly_avg.index, y=monthly_avg.values, ax=ax)
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Price (₹/quintal)')
            ax.set_title(f'Monthly Average Price for {selected_crop} in {selected_city}')
            plt.xticks(rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    except FileNotFoundError:
        st.error("Required datasets not found. Please upload the necessary CSV files.")
        
        # Option to upload data
        st.subheader("Upload Crop Price Data")
        uploaded_file = st.file_uploader("Upload crop_prices.csv", type="csv")
        
        if uploaded_file is not None:
            # Save uploaded file
            df = pd.read_csv(uploaded_file)
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/crop_prices.csv', index=False)
            st.success("File uploaded successfully. Please refresh the page.")