import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class Visualizer:
    @staticmethod
    def plot_price_trends(df, crop_name, city=None, figsize=(12, 6)):
        """Plot price trends for a specific crop and optional city."""
        plt.figure(figsize=figsize)
        
        # Filter data
        filtered_df = df[df['crop_name'] == crop_name].copy()
        if city:
            filtered_df = filtered_df[filtered_df['city'] == city]
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        
        # Sort by date
        filtered_df = filtered_df.sort_values('date')
        
        # Group by date and calculate mean price if multiple cities
        if city is None:
            grouped_df = filtered_df.groupby('date')['price'].mean().reset_index()
            plt.plot(grouped_df['date'], grouped_df['price'], marker='o', linestyle='-')
            plt.title(f'Average Price Trend for {crop_name} Across All Cities')
        else:
            plt.plot(filtered_df['date'], filtered_df['price'], marker='o', linestyle='-')
            plt.title(f'Price Trend for {crop_name} in {city}')
        
        plt.xlabel('Date')
        plt.ylabel('Price (₹/quintal)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_crop_comparison(df, crops, city, metric='price', figsize=(12, 6)):
        """Compare multiple crops for a specific city based on a metric."""
        plt.figure(figsize=figsize)
        
        # Filter data
        filtered_df = df[df['city'] == city].copy()
        filtered_df = filtered_df[filtered_df['crop_name'].isin(crops)]
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        
        # Sort by date
        filtered_df = filtered_df.sort_values('date')
        
        # Plot each crop
        for crop in crops:
            crop_data = filtered_df[filtered_df['crop_name'] == crop]
            plt.plot(crop_data['date'], crop_data[metric], marker='o', linestyle='-', label=crop)
        
        plt.title(f'Comparison of {metric.capitalize()} for Selected Crops in {city}')
        plt.xlabel('Date')
        plt.ylabel(f'{metric.capitalize()} ({"₹/quintal" if metric == "price" else "quintals"})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_seasonal_pattern(df, crop_name, city=None, figsize=(12, 6)):
        """Plot seasonal patterns for a specific crop."""
        plt.figure(figsize=figsize)
        
        # Filter data
        filtered_df = df[df['crop_name'] == crop_name].copy()
        if city:
            filtered_df = filtered_df[filtered_df['city'] == city]
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        
        # Extract month and year
        filtered_df['month'] = filtered_df['date'].dt.month_name()
        filtered_df['year'] = filtered_df['date'].dt.year
        
        # Calculate monthly averages
        monthly_avg = filtered_df.groupby('month')['price'].mean().reset_index()
        
        # Set correct month order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg['month'] = pd.Categorical(monthly_avg['month'], categories=month_order, ordered=True)
        monthly_avg = monthly_avg.sort_values('month')
        
        # Plot monthly averages
        sns.barplot(x='month', y='price', data=monthly_avg)
        
        if city:
            plt.title(f'Monthly Average Price for {crop_name} in {city}')
        else:
            plt.title(f'Monthly Average Price for {crop_name} Across All Cities')
        
        plt.xlabel('Month')
        plt.ylabel('Average Price (₹/quintal)')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_crop_recommendations(recommendations, figsize=(12, 6)):
        """Visualize crop recommendations."""
        plt.figure(figsize=figsize)
        
        # Plot suitability scores
        sns.barplot(y='crop_name', x='suitability_score', data=recommendations.head(10))
        plt.title('Top 10 Recommended Crops by Suitability Score')
        plt.xlabel('Suitability Score (0-100)')
        plt.ylabel('Crop')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def create_crop_recommendation_table(recommendations, soil_data):
        """Create a styled table for crop recommendations."""
        # Select relevant columns
        display_df = recommendations[['crop_name', 'suitability_score', 'profit_margin', 'growth_duration_days']]
        
        # Rename columns for better display
        display_df.columns = ['Crop', 'Suitability Score', 'Expected Profit (%)', 'Growth Duration (days)']
        
        # Add soil compatibility note
        if 'ideal_pH' in recommendations.columns and 'pH' in soil_data:
            display_df['Soil Compatibility'] = recommendations.apply(
                lambda x: 'Excellent' if abs(x['ideal_pH'] - soil_data['pH']) < 0.5 else
                         ('Good' if abs(x['ideal_pH'] - soil_data['pH']) < 1.0 else 'Fair'),
                axis=1
            )
        
        # Round numerical columns
        for col in ['Suitability Score', 'Expected Profit (%)']:
            display_df[col] = display_df[col].round(1)
        
        return display_df