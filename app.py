import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import components
from components.price_prediction import render_price_prediction
from components.crop_recommendation import render_crop_recommendation
from components.marketplace import render_marketplace

# Set page configuration
st.set_page_config(
    page_title="Crop Market Intelligence System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# User authentication
def authenticate():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.username = None
        
    if not st.session_state.logged_in:
        st.sidebar.title("Login")
        
        # Login options
        login_option = st.sidebar.radio("Select login type:", ["Farmer", "Company"])
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            # In a real app, verify credentials against a database
            # For this demo, any non-empty username/password is accepted
            if username and password:
                st.session_state.logged_in = True
                st.session_state.user_type = login_option
                st.session_state.username = username
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid username or password")
                
        # Registration option
        st.sidebar.markdown("---")
        st.sidebar.subheader("New User?")
        if st.sidebar.button("Register"):
            st.sidebar.info("Registration functionality would be implemented here")
    else:
        st.sidebar.title(f"Welcome, {st.session_state.username}")
        st.sidebar.write(f"User Type: {st.session_state.user_type}")
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_type = None
            st.session_state.username = None
            st.experimental_rerun()
    
    return st.session_state.logged_in, st.session_state.user_type

# Main application
def main():
    # Display header
    st.title("🌾 Crop Market Intelligence System")
    
    # Check authentication
    logged_in, user_type = authenticate()
    
    if logged_in:
        # Navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select a page:",
            ["Crop Price Prediction", "Crop Recommendation", "Marketplace"]
        )
        
        # Render selected page
        if page == "Crop Price Prediction":
            render_price_prediction()
        elif page == "Crop Recommendation":
            render_crop_recommendation()
        elif page == "Marketplace":
            render_marketplace(user_type)
    else:
        # Welcome page for non-logged in users
        st.markdown("""
        ## Welcome to Crop Market Intelligence System
        
        This platform helps farmers make informed decisions about crop selection and understand market trends.
        
        ### Key Features:
        - **Crop Price Prediction**: Forecast prices for the next 5-6 months
        - **Crop Recommendation**: Get personalized crop suggestions based on soil conditions and market trends
        - **Marketplace**: Connect with companies looking to purchase specific crops
        
        Please login to access these features.
        """)
        
        # Display sample charts to demonstrate functionality
        st.subheader("Sample Price Prediction")
        fig, ax = plt.subplots(figsize=(10, 6))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
        wheat_prices = [2500, 2550, 2600, 2800, 2750, 2650, 2700, 2900, 3000]
        rice_prices = [3200, 3250, 3300, 3400, 3450, 3500, 3550, 3600, 3700]
        
        ax.plot(months, wheat_prices, marker='o', linewidth=2, label='Wheat')
        ax.plot(months, rice_prices, marker='s', linewidth=2, label='Rice')
        ax.set_xlabel('Month')
        ax.set_ylabel('Price (₹/quintal)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Sample Crop Price Trends')
        st.pyplot(fig)

if __name__ == "__main__":
    main()