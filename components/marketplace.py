import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

def render_marketplace(user_type):
    st.header("Crop Marketplace")
    
    # Create marketplace data file if it doesn't exist
    marketplace_file = 'data/marketplace.csv'
    if not os.path.exists(marketplace_file):
        marketplace_df = pd.DataFrame(columns=[
            'company_name', 'crop_name', 'quantity_required', 'price_offered',
            'quality_requirements', 'delivery_date', 'contact_info', 'location', 'post_date'
        ])
        os.makedirs('data', exist_ok=True)
        marketplace_df.to_csv(marketplace_file, index=False)
    
    # Load marketplace data
    marketplace_df = pd.read_csv(marketplace_file)
    
    # Different views based on user type
    if user_type == "Company":
        st.subheader("Post Your Crop Requirements")
        
        with st.form("crop_requirement_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                company_name = st.text_input("Company Name", value=st.session_state.username)
                crop_name = st.text_input("Crop Name")
                quantity = st.number_input("Quantity Required (quintals)", min_value=1, value=100)
                price = st.number_input("Price Offered (₹/quintal)", min_value=1, value=2000)
            
            with col2:
                quality = st.text_area("Quality Requirements", "Grade A, Moisture content below 14%")
                delivery_date = st.date_input("Delivery Date")
                contact = st.text_input("Contact Information", "Email or Phone Number")
                location = st.text_input("Location")
            
            submit_button = st.form_submit_button("Post Requirement")
            
            if submit_button:
                # Add new requirement to the dataframe
                new_requirement = {
                    'company_name': company_name,
                    'crop_name': crop_name,
                    'quantity_required': quantity,
                    'price_offered': price,
                    'quality_requirements': quality,
                    'delivery_date': delivery_date.strftime('%Y-%m-%d'),
                    'contact_info': contact,
                    'location': location,
                    'post_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                marketplace_df = pd.concat([marketplace_df, pd.DataFrame([new_requirement])], ignore_index=True)
                marketplace_df.to_csv(marketplace_file, index=False)
                st.success("Requirement posted successfully!")
        
        # View and manage posted requirements
        st.subheader("Your Posted Requirements")
        company_requirements = marketplace_df[marketplace_df['company_name'] == st.session_state.username]
        
        if not company_requirements.empty:
            for i, req in company_requirements.iterrows():
                with st.expander(f"{req['crop_name']} - {req['quantity_required']} quintals"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Price Offered:** ₹{req['price_offered']}/quintal")
                        st.write(f"**Quality:** {req['quality_requirements']}")
                        st.write(f"**Delivery Date:** {req['delivery_date']}")
                    
                    with col2:
                        st.write(f"**Location:** {req['location']}")
                        st.write(f"**Posted on:** {req['post_date']}")
                        st.write(f"**Contact:** {req['contact_info']}")
                    
                    if st.button(f"Delete this requirement", key=f"delete_{i}"):
                        marketplace_df = marketplace_df.drop(i)
                        marketplace_df.to_csv(marketplace_file, index=False)
                        st.success("Requirement deleted successfully!")
                        st.experimental_rerun()
        else:
            st.info("You haven't posted any requirements yet.")
    
    elif user_type == "Farmer":
        st.subheader("Browse Crop Requirements")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            # Load crop names from marketplace
            all_crops = sorted(marketplace_df['crop_name'].unique()) if not marketplace_df.empty else []
            selected_crop = st.selectbox("Filter by Crop", ["All"] + all_crops)
        
        with col2:
            # Load locations from marketplace
            all_locations = sorted(marketplace_df['location'].unique()) if not marketplace_df.empty else []
            selected_location = st.selectbox("Filter by Location", ["All"] + all_locations)
        
        # Apply filters
        filtered_df = marketplace_df.copy()
        
        if selected_crop != "All":
            filtered_df = filtered_df[filtered_df['crop_name'] == selected_crop]
        
        if selected_location != "All":
            filtered_df = filtered_df[filtered_df['location'] == selected_location]
        
        # Sort options
        sort_by = st.selectbox("Sort by", ["Price (High to Low)", "Price (Low to High)", "Most Recent"])
        
        if sort_by == "Price (High to Low)":
            filtered_df = filtered_df.sort_values(by='price_offered', ascending=False)
        elif sort_by == "Price (Low to High)":
            filtered_df = filtered_df.sort_values(by='price_offered', ascending=True)
        else:  # Most Recent
            filtered_df = filtered_df.sort_values(by='post_date', ascending=False)
        
        # Display requirements
        if not filtered_df.empty:
            st.write(f"Showing {len(filtered_df)} requirements")
            
            for i, req in filtered_df.iterrows():
                with st.expander(f"{req['company_name']} needs {req['crop_name']} - ₹{req['price_offered']}/quintal"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Company:** {req['company_name']}")
                        st.write(f"**Quantity:** {req['quantity_required']} quintals")
                        st.write(f"**Price:** ₹{req['price_offered']}/quintal")
                        st.write(f"**Quality:** {req['quality_requirements']}")
                    
                    with col2:
                        st.write(f"**Location:** {req['location']}")
                        st.write(f"**Delivery by:** {req['delivery_date']}")
                        st.write(f"**Posted on:** {req['post_date']}")
                        st.write(f"**Contact:** {req['contact_info']}")