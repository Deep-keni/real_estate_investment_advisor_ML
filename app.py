import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.predict import predict_investment, predict_price

st.title("🏠 Real Estate Investment Advisor")
st.subheader("Predict Property Profitability & Future Value")

st.markdown("---")

st.header("📋 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
    size = st.number_input("Size (SqFt)", min_value=100, max_value=10000, value=1000)
    price = st.number_input("Price (in Lakhs)", min_value=1, max_value=500, value=50)
    price_per_sqft = price / size

with col2:
    floor_no = st.number_input("Floor No", min_value=0, max_value=50, value=1)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=5)
    age = st.number_input("Age of Property", min_value=0, max_value=50, value=5)
    parking = st.selectbox("Parking Space", [0, 1])

with col3:
    schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=3)
    hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=10, value=3)
    amenities = st.number_input("Amenities Count", min_value=0, max_value=10, value=3)

st.markdown("---")

col4, col5 = st.columns(2)

with col4:
    furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
    furnished_encoded = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}[furnished]
    
    transport = st.selectbox("Public Transport", ["Low", "Medium", "High"])
    transport_encoded = {"Low": 0, "Medium": 1, "High": 2}[transport]
    
    security = st.selectbox("Security", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col5:
    availability = st.selectbox("Availability Status", options=[0, 1, 2], 
                                format_func=lambda x: ["Available", "Under Construction", "Sold"][x])
    
    property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
    facing = st.selectbox("Facing", ["East", "North", "South", "West"])
    owner_type = st.selectbox("Owner Type", ["Agent", "Builder", "Owner"])

st.markdown("---")

if st.button("🔍 Predict"):
    
    # OHE columns for property type
    prop_independent = 1 if property_type == "Independent House" else 0
    prop_villa = 1 if property_type == "Villa" else 0
    
    # OHE columns for facing
    facing_north = 1 if facing == "North" else 0
    facing_south = 1 if facing == "South" else 0
    facing_west = 1 if facing == "West" else 0
    
    # OHE columns for owner type
    owner_builder = 1 if owner_type == "Builder" else 0
    owner_owner = 1 if owner_type == "Owner" else 0

    # input for classification (has Price_in_Lakhs, no Price_per_SqFt)
    input_clf = pd.DataFrame([{
        'D': 0,
        'State': 0,
        'City': 0,
        'BHK': bhk,
        'Size_in_SqFt': size,
        'Price_in_Lakhs': price,
        'Furnished_Status': furnished_encoded,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age,
        'Nearby_Schools': schools,
        'Nearby_Hospitals': hospitals,
        'Public_Transport_Accessibility': transport_encoded,
        'Parking_Space': parking,
        'Security': security,
        'Availability_Status': availability,
        'Property_Type_Independent House': prop_independent,
        'Property_Type_Villa': prop_villa,
        'Facing_North': facing_north,
        'Facing_South': facing_south,
        'Facing_West': facing_west,
        'Owner_Type_Builder': owner_builder,
        'Owner_Type_Owner': owner_owner,
        'Amenities_Count': amenities
    }])

    # input for regression (has Price_per_SqFt, no Price_in_Lakhs)
    input_rgr = pd.DataFrame([{
        'D': 0,
        'State': 0,
        'City': 0,
        'BHK': bhk,
        'Size_in_SqFt': size,
        'Price_per_SqFt': price_per_sqft,
        'Furnished_Status': furnished_encoded,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age,
        'Nearby_Schools': schools,
        'Nearby_Hospitals': hospitals,
        'Public_Transport_Accessibility': transport_encoded,
        'Parking_Space': parking,
        'Security': security,
        'Availability_Status': availability,
        'Property_Type_Independent House': prop_independent,
        'Property_Type_Villa': prop_villa,
        'Facing_North': facing_north,
        'Facing_South': facing_south,
        'Facing_West': facing_west,
        'Owner_Type_Builder': owner_builder,
        'Owner_Type_Owner': owner_owner,
        'Amenities_Count': amenities
    }])

    # predictions
    pred, prob = predict_investment(input_clf)
    future_price = predict_price(input_rgr)

    # show results
    st.markdown("## 🎯 Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if pred == 1:
            st.success(f"✅ Good Investment! Confidence: {prob[1]*100:.1f}%")
        else:
            st.error(f"❌ Not a Good Investment. Confidence: {prob[0]*100:.1f}%")
    
    with col2:
        st.info(f"📈 Estimated Price after 5 Years: ₹{future_price:.2f} Lakhs")

    
st.markdown("---")

st.header("📊 EDA Insights")

st.subheader("Price Distribution")
st.image("visuals/price_distribution.png")

st.subheader("Average Price by State & City")
st.image("visuals/avg_price_by_state_city.png")

st.subheader("Correlation Heatmap")
st.image("visuals/correlation_heatmap.png")

st.subheader("Good Investment Distribution")
st.image("visuals/good_investment_public_transport.png")

st.subheader("BHK vs Average Price")
st.image("visuals/bhk_vs_price.png")