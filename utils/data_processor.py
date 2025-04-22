import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataProcessor:
    @staticmethod
    def validate_csv_format(file, expected_columns):
        """Validate if the uploaded CSV file has the required columns."""
        try:
            df = pd.read_csv(file)
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                return False, f"Missing columns: {', '.join(missing_columns)}"
            
            return True, df
        except Exception as e:
            return False, f"Error reading CSV file: {str(e)}"
    
    @staticmethod
    def generate_sample_data():
        """Generate sample data for demonstration if no data is uploaded."""
        # Generate crop prices data
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
        crops = ['Rice', 'Wheat', 'Maize', 'Potato', 'Tomato', 'Onion', 'Cotton', 'Sugarcane']
        
        # Generate dates for the past 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        dates = pd.date_range(start=start_date, end=end_date, freq='15D')
        
        # Generate crop prices data
        crop_prices_data = []
        
        for city in cities:
            for crop in crops:
                # Base price for this crop in this city
                base_price = np.random.randint(1500, 5000)
                
                for date in dates:
                    # Add some randomness and seasonal variation
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
                    random_factor = np.random.uniform(0.9, 1.1)
                    price = base_price * seasonal_factor * random_factor
                    
                    # Quantity sold
                    quantity = np.random.randint(500, 2000)
                    
                    crop_prices_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'crop_name': crop,
                        'city': city,
                        'price': round(price, 2),
                        'quantity_sold': quantity
                    })
        
        crop_prices_df = pd.DataFrame(crop_prices_data)
        
        # Generate soil data
        soil_data = []
        
        soil_types = ['Sandy', 'Loamy', 'Clay', 'Silt', 'Peaty', 'Chalky', 'Laterite']
        
        for city in cities:
            # State corresponding to city (simplified)
            state_map = {
                'Mumbai': 'Maharashtra',
                'Delhi': 'Delhi',
                'Bangalore': 'Karnataka',
                'Chennai': 'Tamil Nadu',
                'Kolkata': 'West Bengal',
                'Hyderabad': 'Telangana',
                'Pune': 'Maharashtra',
                'Ahmedabad': 'Gujarat'
            }
            
            soil_data.append({
                'city': city,
                'state': state_map[city],
                'soil_type': np.random.choice(soil_types),
                'pH': round(np.random.uniform(5.5, 8.0), 1),
                'nitrogen': np.random.randint(50, 120),
                'phosphorus': np.random.randint(30, 100),
                'potassium': np.random.randint(20, 80),
                'temperature': np.random.randint(22, 35),
                'humidity': np.random.randint(50, 90),
                'rainfall': np.random.randint(70, 300)
            })
        
        soil_df = pd.DataFrame(soil_data)
        
        # Generate crop details data
        crop_details_data = []
        
        for crop in crops + ['Chickpea', 'Lentil', 'Soybean', 'Mustard', 'Groundnut']:
            growth_duration = np.random.randint(90, 180)
            water_req = np.random.randint(300, 1200)
            
            crop_details_data.append({
                'crop_name': crop,
                'growth_duration_days': growth_duration,
                'water_requirement': water_req,
                'nitrogen_requirement': np.random.randint(40, 120),
                'phosphorus_requirement': np.random.randint(20, 80),
                'potassium_requirement': np.random.randint(20, 60),
                'ideal_temperature': np.random.randint(18, 35),
                'ideal_pH': round(np.random.uniform(5.5, 7.5), 1),
                'profit_margin': np.random.randint(15, 40)
            })
        
        crop_details_df = pd.DataFrame(crop_details_data)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save data to CSV files
        crop_prices_df.to_csv('data/crop_prices.csv', index=False)
        soil_df.to_csv('data/soil_data.csv', index=False)
        crop_details_df.to_csv('data/crop_details.csv', index=False)
        
        return {
            'crop_prices': crop_prices_df,
            'soil_data': soil_df,
            'crop_details': crop_details_df
        }
    
    @staticmethod
    def preprocess_time_series(df, date_column='date', target_column='price'):
        """Preprocess time series data for modeling."""
        # Convert date to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Extract date features
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        
        # Create lagged features
        df[f'{target_column}_lag1'] = df[target_column].shift(1)
        df[f'{target_column}_lag2'] = df[target_column].shift(2)
        df[f'{target_column}_lag3'] = df[target_column].shift(3)
        
        # Create rolling mean features
        df[f'{target_column}_ma7'] = df[target_column].rolling(window=7, min_periods=1).mean()
        df[f'{target_column}_ma30'] = df[target_column].rolling(window=30, min_periods=1).mean()
        
        # Fill missing values
        df = df.fillna(method='bfill')
        if df.isna().sum().sum() > 0:
            df = df.fillna(df.mean())
        
        return df