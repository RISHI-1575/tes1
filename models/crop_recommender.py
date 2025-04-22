import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

class CropRecommender:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.crop_list = [
            'Rice', 'Wheat', 'Maize', 'Chickpea', 'Kidney Beans', 'Pigeon Peas',
            'Moth Beans', 'Mung Bean', 'Black Gram', 'Lentil', 'Pomegranate',
            'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple',
            'Orange', 'Papaya', 'Coconut', 'Cotton', 'Jute', 'Coffee'
        ]
        
    def train(self, data=None):
        """
        Train the crop recommendation model.
        If no data is provided, uses synthetic data for demo purposes.
        In a real application, you would use actual field data.
        """
        if data is None:
            # Create synthetic data for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic data based on general crop requirements
            data = pd.DataFrame({
                'N': np.random.randint(0, 150, n_samples),
                'P': np.random.randint(0, 150, n_samples),
                'K': np.random.randint(0, 150, n_samples),
                'temperature': np.random.uniform(10, 40, n_samples),
                'humidity': np.random.uniform(20, 100, n_samples),
                'pH': np.random.uniform(3, 10, n_samples),
                'rainfall': np.random.uniform(50, 300, n_samples)
            })
            
            # Assign crops based on general requirements
            def assign_crop(row):
                # This is a simplified logic for demonstration
                if (row['temperature'] > 22 and row['temperature'] < 32 and 
                    row['humidity'] > 80 and row['rainfall'] > 200):
                    return 'Rice'
                elif (row['temperature'] > 15 and row['temperature'] < 25 and 
                      row['rainfall'] < 100):
                    return 'Wheat'
                elif (row['N'] > 80 and row['P'] > 60 and row['K'] > 40 and 
                      row['temperature'] > 20 and row['temperature'] < 30):
                    return 'Maize'
                elif (row['temperature'] > 25 and row['rainfall'] > 150 and 
                      row['pH'] > 6 and row['pH'] < 7.5):
                    return 'Banana'
                elif (row['temperature'] > 20 and row['temperature'] < 35 and 
                      row['rainfall'] > 100 and row['rainfall'] < 200):
                    return 'Cotton'
                else:
                    return np.random.choice(self.crop_list)
                
            data['label'] = data.apply(assign_crop, axis=1)
        
        # Prepare features and target
        X = data[['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']]
        y = data['label']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        print("Crop recommendation model trained successfully.")
        return self.model
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump((self.model, self.scaler), f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            self.model, self.scaler = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return self.model
    
    def recommend(self, input_values, crop_details, crop_prices, selected_city):
        """
        Recommend crops based on soil conditions, market trends, and seasonal factors.
        Returns DataFrame with recommended crops and their scores.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare input for prediction
        input_df = pd.DataFrame({
            'N': [input_values['N']],
            'P': [input_values['P']],
            'K': [input_values['K']],
            'temperature': [input_values['temperature']],
            'humidity': [input_values['humidity']],
            'pH': [input_values['pH']],
            'rainfall': [input_values['rainfall']]
        })
        
        # Scale input
        input_scaled = self.scaler.transform(input_df)
        
        # Get probability scores for each crop
        crop_probabilities = self.model.predict_proba(input_scaled)[0]
        
        # Create a dataframe with crop names and their suitability scores
        crop_names = self.model.classes_
        recommendations = pd.DataFrame({
            'crop_name': crop_names,
            'suitability_score': crop_probabilities * 100  # Convert to percentage
        })
        
        # Sort by suitability score
        recommendations = recommendations.sort_values('suitability_score', ascending=False).reset_index(drop=True)
        
        # Merge with crop details
        if crop_details is not None:
            recommendations = pd.merge(recommendations, crop_details, on='crop_name', how='left')
        
        # Calculate market score based on price trends if data is available
        if crop_prices is not None:
            # Filter for the selected city
            city_prices = crop_prices[crop_prices['city'] == selected_city]
            
            # Calculate market score for each crop
            market_scores = []
            
            for crop in recommendations['crop_name']:
                crop_price_data = city_prices[city_prices['crop_name'] == crop]
                
                if not crop_price_data.empty:
                    # Convert date to datetime and sort
                    crop_price_data['date'] = pd.to_datetime(crop_price_data['date'])
                    crop_price_data = crop_price_data.sort_values('date')
                    
                    # Calculate price trend (percentage change in last period)
                    if len(crop_price_data) > 1:
                        latest_price = crop_price_data['price'].iloc[-1]
                        previous_price = crop_price_data['price'].iloc[-2]
                        price_trend = ((latest_price - previous_price) / previous_price) * 100
                    else:
                        price_trend = 0
                    
                    market_scores.append(price_trend)
                else:
                    market_scores.append(0)
            
            recommendations['market_trend'] = market_scores
            
            # Adjust profit margin based on market trend
            if 'profit_margin' in recommendations.columns:
                recommendations['adjusted_profit'] = recommendations['profit_margin'] + (recommendations['market_trend'] * 0.5)
                recommendations['profit_margin'] = recommendations['adjusted_profit'].clip(lower=0)
                recommendations.drop('adjusted_profit', axis=1, inplace=True)
        
        # Get top 10 recommendations
        top_recommendations = recommendations.head(10)
        
        return top_recommendations