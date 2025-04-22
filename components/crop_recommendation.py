import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import the crop recommender model
from models.crop_recommender import CropRecommender

def render_crop_recommendation():
    st.header("Crop Recommendation System")
    st.write("Get personalized crop recommendations based on your location, soil conditions, and market trends.")
    
    try:
        # Load datasets
        soil_data = pd.read_csv('data/soil_data.csv')
        crop_details = pd.read_csv('data/crop_details.csv')
        crop_prices = pd.read_csv('data/crop_prices.csv')
        
        # Get unique cities
        cities = sorted(soil_data['city'].unique())
        
        # User input
        selected_city = st.selectbox("Select Your City", cities)
        
        # Get soil data for selected city
        city_soil = soil_data[soil_data['city'] == selected_city].iloc[0]
        
        # Display soil characteristics
        st.subheader("Soil Characteristics for " + selected_city)
        soil_cols = ['soil_type', 'pH', 'nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'rainfall']
        soil_df = pd.DataFrame([city_soil[soil_cols]], columns=soil_cols)
        st.dataframe(soil_df)
        
        # Allow user to adjust soil parameters
        st.subheader("Adjust Soil Parameters (Optional)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nitrogen = st.slider("Nitrogen (kg/ha)", 0, 150, int(city_soil['nitrogen']))
            phosphorus = st.slider("Phosphorus (kg/ha)", 0, 150, int(city_soil['phosphorus']))
        
        with col2:
            potassium = st.slider("Potassium (kg/ha)", 0, 150, int(city_soil['potassium']))
            soil_ph = st.slider("Soil pH", 0.0, 14.0, float(city_soil['pH']))
        
        with col3:
            rainfall = st.slider("Annual Rainfall (mm)", 0, 3000, int(city_soil['rainfall']))
            temperature = st.slider("Temperature (°C)", 0, 50, int(city_soil['temperature']))
        
        # Create inputs for the recommendation model
        inputs = {
            'N': nitrogen,
            'P': phosphorus,
            'K': potassium,
            'pH': soil_ph,
            'rainfall': rainfall,
            'temperature': temperature,
            'humidity': city_soil['humidity']
        }
        
        # Create recommender instance
        recommender = CropRecommender()
        
        # Check if model exists, if not train a new one
        model_path = 'models/saved/crop_recommender_model.pkl'
        
        if os.path.exists(model_path):
            # Load existing model
            recommender.load_model(model_path)
        else:
            st.info("Training recommendation model for the first time... This may take a moment.")
            # In a real app, would train with combined soil and crop data
            recommender.train()
            
            # Save model
            os.makedirs('models/saved', exist_ok=True)
            recommender.save_model(model_path)
        
        # Make recommendations
        if st.button("Get Crop Recommendations"):
            recommended_crops = recommender.recommend(inputs, crop_details, crop_prices, selected_city)
            
            if recommended_crops is not None and not recommended_crops.empty:
                st.subheader("Recommended Crops")
                
                # Format the dataframe for display
                display_df = recommended_crops[['crop_name', 'suitability_score', 'profit_margin', 'growth_duration_days']]
                display_df.columns = ['Crop', 'Suitability Score (0-100)', 'Expected Profit (%)', 'Growth Duration (days)']
                
                # Add color coding based on suitability score
                def highlight_suitability(val):
                    if val > 80:
                        return 'background-color: #8eff8e'  # Green
                    elif val > 60:
                        return 'background-color: #fff78e'  # Yellow
                    else:
                        return 'background-color: #ff8e8e'  # Red
                
                # Apply styling
                styled_df = display_df.style.applymap(
                    highlight_suitability, 
                    subset=['Suitability Score (0-100)']
                )
                
                st.dataframe(styled_df)
                
                # Visualization of recommendations
                st.subheader("Comparison of Recommended Crops")
                
                # Plot suitability vs profit
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=recommended_crops, 
                    x='suitability_score', 
                    y='profit_margin',
                    size='growth_duration_days',
                    sizes=(50, 400),
                    hue='crop_name',
                    ax=ax
                )
                ax.set_xlabel('Suitability Score')
                ax.set_ylabel('Expected Profit (%)')
                ax.set_title('Crop Suitability vs. Profit Potential')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Plot growth duration
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=recommended_crops,
                    y='crop_name',
                    x='growth_duration_days',
                    ax=ax
                )
                ax.set_xlabel('Growth Duration (days)')
                ax.set_ylabel('Crop')
                ax.set_title('Growth Duration by Crop')
                ax.grid(True, axis='x', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Additional insights
                st.subheader("Market Insights")
                
                # Get latest price trends for top recommended crops
                top_crops = recommended_crops['crop_name'].tolist()[:3]
                
                for crop in top_crops:
                    crop_price_data = crop_prices[
                        (crop_prices['crop_name'] == crop) & 
                        (crop_prices['city'] == selected_city)
                    ]
                    
                    if not crop_price_data.empty:
                        # Convert date to datetime and sort
                        crop_price_data['date'] = pd.to_datetime(crop_price_data['date'])
                        crop_price_data = crop_price_data.sort_values('date')
                        
                        # Plot price trend
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(
                            crop_price_data['date'], 
                            crop_price_data['price'],
                            marker='o',
                            linestyle='-'
                        )
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Price (₹/quintal)')
                        ax.set_title(f'{crop} Price Trend in {selected_city}')
                        plt.xticks(rotation=45)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
            else:
                st.warning("No suitable crops found for the given conditions. Try adjusting the soil parameters.")
    
    except FileNotFoundError:
        st.error("Required datasets not found. Please upload the necessary CSV files.")
        
        # Option to upload data
        st.subheader("Upload Required Data")
        
        upload_cols = st.columns(3)
        
        with upload_cols[0]:
            soil_file = st.file_uploader("Upload soil_data.csv", type="csv")
            if soil_file is not None:
                soil_data = pd.read_csv(soil_file)
                os.makedirs('data', exist_ok=True)
                soil_data.to_csv('data/soil_data.csv', index=False)
                st.success("Soil data uploaded successfully.")
        
        with upload_cols[1]:
            crop_details_file = st.file_uploader("Upload crop_details.csv", type="csv")
            if crop_details_file is not None:
                crop_details = pd.read_csv(crop_details_file)
                os.makedirs('data', exist_ok=True)
                crop_details.to_csv('data/crop_details.csv', index=False)
                st.success("Crop details uploaded successfully.")
        
        with upload_cols[2]:
            prices_file = st.file_uploader("Upload crop_prices.csv", type="csv")
            if prices_file is not None:
                prices = pd.read_csv(prices_file)
                os.makedirs('data', exist_ok=True)
                prices.to_csv('data/crop_prices.csv', index=False)
                st.success("Crop prices uploaded successfully.")
        
        if soil_file is not None and crop_details_file is not None and prices_file is not None:
            st.success("All required files uploaded. Please refresh the page.")