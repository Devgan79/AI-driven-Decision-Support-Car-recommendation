import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load Data (Assume preprocessing already done in a CSV)
df = pd.read_csv("Dataset/car_price_dataset.csv")

# Preprocess (One-Hot Encoding + Normalization)
categorical_columns = ["Brand", "Model", "Fuel_Type", "Transmission"]
df_encoded = pd.get_dummies(df, columns=categorical_columns)

numerical_columns = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count", "Price"]
scaler = MinMaxScaler()
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

# Similarity Matrix
features_for_similarity = df_encoded.drop(columns=["Price"])  # Exclude Price
similarity_matrix = cosine_similarity(features_for_similarity)

st.title("🚗 Car Recommendation System")

brand = st.selectbox("Select Brand", df["Brand"].unique())
model = st.selectbox("Select Model", df[df["Brand"] == brand]["Model"].unique())
year = st.slider("Select Year", int(df["Year"].min()), int(df["Year"].max()), 2020)
engine_size = st.slider("Engine Size (Litres)", float(df["Engine_Size"].min()), float(df["Engine_Size"].max()), 1.6)
mileage = st.slider("Mileage (km)", int(df["Mileage"].min()), int(df["Mileage"].max()), 50000)
fuel_type = st.selectbox("Fuel Type", df["Fuel_Type"].unique())
transmission = st.selectbox("Transmission", df["Transmission"].unique())
doors = st.slider("Number of Doors", int(df["Doors"].min()), int(df["Doors"].max()), 4)
owner_count = st.slider("Number of Previous Owners", int(df["Owner_Count"].min()), int(df["Owner_Count"].max()), 1)
price = st.slider("Expected Price", int(df["Price"].min()), int(df["Price"].max()), 20000)

# Create User Preference Vector
user_preferences = {
    "Year": year,
    "Engine_Size": engine_size,
    "Mileage": mileage,
    "Doors": doors,
    "Owner_Count": owner_count,
    "Price": price
}

# One-hot encode user preference
for col in categorical_columns:
    for val in df[col].unique():
        user_preferences[f"{col}_{val}"] = 1 if (col == "Brand" and val == brand) or \
                                                 (col == "Model" and val == model) or \
                                                 (col == "Fuel_Type" and val == fuel_type) or \
                                                 (col == "Transmission" and val == transmission) else 0

user_df = pd.DataFrame([user_preferences])

user_df = user_df.reindex(columns=df_encoded.columns, fill_value=0)
user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])
similarity_scores = cosine_similarity(user_df, df_encoded)[0]

# Get Top 5 Recommendations/ you can  Channge if you like 
top_indices = np.argsort(similarity_scores)[-5:][::-1]
recommended_cars = df.iloc[top_indices][["Brand", "Model", "Year", "Price", "Fuel_Type", "Transmission"]]

# Display Recommendations
st.subheader("Top Recommended Cars for You:")
st.write(recommended_cars)



# TO RUN THIS TYPE THIS IN THE TERMINAL streamlit run main.py
