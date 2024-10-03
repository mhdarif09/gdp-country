import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st

# Set page title and layout
st.set_page_config(page_title="GDP Prediction Dashboard", page_icon=":bar_chart:")

# Load GDP and Interest Rate Data
@st.cache_data
def load_gdp_data():
    gdp_file = 'data/gdp_data.csv'
    gdp_data = pd.read_csv(gdp_file)
    return gdp_data

@st.cache_data
def load_interest_rate_data():
    interest_rate_file = 'data/API_FR.INR.RINR_DS2_en_csv_v2_5728810.csv'
    interest_rate_data = pd.read_csv(interest_rate_file, skiprows=4)
    return interest_rate_data

# Load both datasets
gdp_df = load_gdp_data()
interest_rate_df = load_interest_rate_data()

# Show data preview
st.write("GDP Data:", gdp_df.head())
st.write("Interest Rate Data:", interest_rate_df.head())

# Reshape interest rate data (if needed)
MIN_YEAR = 1960
MAX_YEAR = 2022
interest_rate_df = interest_rate_df.melt(id_vars=['Country Name'], value_vars=[str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)], var_name='Year', value_name='Interest Rate')
interest_rate_df['Year'] = pd.to_numeric(interest_rate_df['Year'])

# Merge GDP and Interest Rate data
merged_data = pd.merge(gdp_df, interest_rate_df, on=['Country Name', 'Year'], how='inner')
st.write("Merged Data:", merged_data.head())

# Select country
countries = merged_data['Country Name'].unique()
selected_country = st.selectbox('Select a Country:', countries, index=0)

# Filter data for the selected country
country_data = merged_data[merged_data['Country Name'] == selected_country]

# Train model for GDP prediction
X = country_data[['Interest Rate']].values  # Interest rate as feature
y = country_data['GDP'].values  # GDP as the target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)

st.write(f'Model Mean Squared Error: {mse:.2f}')

# Predict GDP for the next 5 years
future_years = np.array([2024, 2025, 2026, 2027, 2028])
future_interest_rates = st.slider('Projected Interest Rates for the next 5 years:', min_value=0.0, max_value=15.0, value=[3.0, 3.5, 4.0, 4.5, 5.0], step=0.1)

future_gdp = model.predict(np.array(future_interest_rates).reshape(-1, 1))

# Display future predictions
predicted_gdp_df = pd.DataFrame({
    'Year': future_years,
    'Projected Interest Rate': future_interest_rates,
    'Predicted GDP': future_gdp
})

st.write("Predicted GDP for the next 5 years:", predicted_gdp_df)

# Plot GDP predictions
plt.figure(figsize=(10, 5))
plt.plot(country_data['Year'], country_data['GDP'], label='Historical GDP', color='blue')
plt.plot(future_years, future_gdp, label='Predicted GDP', color='red', marker='o')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title(f'GDP Prediction for {selected_country}')
plt.legend()
st.pyplot(plt)

