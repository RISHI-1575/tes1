import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

class CropPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def _extract_features(self, df):
        """Extract features from the dataframe for model training."""
        df_copy = df.copy()
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Extract date features
        df_copy['year'] = df_copy['date'].dt.year
        df_copy['month'] = df_copy['date'].dt.month
        df_copy['day'] = df_copy['date'].dt.day
        df_copy['day_of_week'] = df_copy['date'].dt.dayofweek
        df_copy['quarter'] = df_copy['date'].dt.quarter
        
        # Create time-based moving averages
        df_copy = df_copy.sort_values('date')
        df_copy['price_lag1'] = df_copy['price'].shift(1)
        df_copy['price_lag2'] = df_copy['price'].shift(2)
        df_copy['price_lag3'] = df_copy['price'].shift(3)
        df_copy['price_ma7'] = df_copy['price'].rolling(window=7, min_periods=1).mean()
        df_copy['price_ma30'] = df_copy['price'].rolling(window=30, min_periods=1).mean()
        
        # Fill NaN values
        df_copy = df_copy.fillna(method='bfill')
        if df_copy.isna().sum().sum() > 0:
            df_copy = df_copy.fillna(df_copy.mean())
        
        # Select features
        features = ['year', 'month', 'day', 'day_of_week', 'quarter', 
                   'price_lag1', 'price_lag2', 'price_lag3', 'price_ma7', 'price_ma30']
        
        # Add quantity sold if available
        if 'quantity_sold' in df_copy.columns:
            features.append('quantity_sold')
        
        X = df_copy[features]
        y = df_copy['price']
        
        return X, y
    
    def train(self, df):
        """Train the price prediction model."""
        # Extract features
        X, y = self._extract_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Model trained. Train R² score: {train_score:.4f}, Test R² score: {test_score:.4f}")
        
        return self.model
    
    def predict(self, df, n_months=6):
        """Predict prices for the next n months."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get the last date in the dataset
        df_copy = df.copy()
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        last_date = df_copy['date'].max()
        last_price = df_copy.loc[df_copy['date'] == last_date, 'price'].values[0]
        
        # Create a dataframe for future dates
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, n_months+1)]
        
        # Initialize with the last known values
        predictions = []
        
        for i, future_date in enumerate(future_dates):
            # Create a feature row for this date
            future_row = pd.DataFrame({
                'year': [future_date.year],
                'month': [future_date.month],
                'day': [future_date.day],
                'day_of_week': [future_date.dayofweek],
                'quarter': [future_date.quarter],
                'price_lag1': [last_price if i == 0 else predictions[-1]],
                'price_lag2': [df_copy['price'].iloc[-1] if i == 0 else (last_price if i == 1 else predictions[-2])],
                'price_lag3': [df_copy['price'].iloc[-2] if i == 0 else (df_copy['price'].iloc[-1] if i == 1 else (last_price if i == 2 else predictions[-3]))]
            })
            
            # Add moving averages
            if i == 0:
                future_row['price_ma7'] = df_copy['price'].tail(7).mean()
                future_row['price_ma30'] = df_copy['price'].tail(30).mean()
            else:
                # Calculate moving averages based on previous predictions and historical data
                historical_prices = list(df_copy['price'].tail(30).values)
                predicted_prices = predictions.copy()
                
                # Calculate MA7
                prices_for_ma7 = historical_prices[-7+i:] + predicted_prices
                future_row['price_ma7'] = np.mean(prices_for_ma7[-7:])
                
                # Calculate MA30
                prices_for_ma30 = historical_prices + predicted_prices
                future_row['price_ma30'] = np.mean(prices_for_ma30[-30:])
            
            # Add quantity_sold if it was in the training data
            if 'quantity_sold' in df_copy.columns:
                future_row['quantity_sold'] = df_copy['quantity_sold'].mean()
            
            # Scale the features
            future_row_scaled = self.scaler.transform(future_row)
            
            # Make prediction
            prediction = self.model.predict(future_row_scaled)[0]
            predictions.append(prediction)
            
            # Update last_price for next iteration
            last_price = prediction
        
        return predictions