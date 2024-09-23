import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP Dashboard with Predictions',
    page_icon=':earth_americas:',
)

# -------------------------------------------------------------------------------
# Helper function to get GDP data

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file."""
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # Pivot the data to create a cleaner format
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert year to numeric
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -------------------------------------------------------------------------------
# Draw the main page

'''
# :earth_americas: GDP Dashboard with Predictions
This dashboard displays GDP data and predicts future GDP based on historical data using **Linear Regression**.
'''

# Add some spacing
''

# Filter countries and years
countries = gdp_df['Country Code'].unique()
selected_countries = st.multiselect(
    'Select countries for prediction:',
    countries,
    ['DEU', 'FRA', 'GBR', 'USA', 'JPN']
)

min_year = gdp_df['Year'].min()
max_year = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Select year range:',
    min_value=min_year,
    max_value=max_year,
    value=[min_year, max_year]
)

# Filter the GDP data based on user selection
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (gdp_df['Year'] >= from_year)
]

# Plot the historical GDP data
st.header('Historical GDP Data')
st.line_chart(filtered_gdp_df, x='Year', y='GDP', color='Country Code')

# -------------------------------------------------------------------------------
# Building and Training the Machine Learning Model

st.header('GDP Prediction')

# Input years for future prediction
future_years = st.slider(
    'Select future years to predict GDP:',
    min_value=max_year + 1,
    max_value=2050,
    value=[2025, 2030, 2040]
)

# Initialize an empty dictionary to store predictions
predictions = {}

# Iterate over selected countries
for country in selected_countries:
    country_data = gdp_df[gdp_df['Country Code'] == country]

    # Prepare the data for the model
    X = country_data[['Year']]
    y = country_data['GDP']

    # Split data into training and test sets (we'll only use training data here)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict GDP for future years
    future_years_array = np.array(future_years).reshape(-1, 1)
    predicted_gdp = model.predict(future_years_array)

    # Store predictions in the dictionary
    predictions[country] = predicted_gdp

    # Display model evaluation (optional)
    y_pred_train = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred_train)
    st.write(f'Mean Squared Error for {country}: {mse:.2f}')

# -------------------------------------------------------------------------------
# Display predictions

st.subheader('GDP Predictions for Selected Years')
cols = st.columns(len(future_years))

for i, year in enumerate(future_years):
    col = cols[i]
    with col:
        st.write(f"**Year: {year}**")
        for country in selected_countries:
            gdp_pred = predictions[country][i] / 1e9  # Convert to billions
            st.metric(f"{country}", f"{gdp_pred:.2f}B")

# Plot predictions
prediction_df = pd.DataFrame({
    'Year': np.repeat(future_years, len(selected_countries)),
    'Country Code': np.tile(selected_countries, len(future_years)),
    'GDP': np.hstack([predictions[country] for country in selected_countries])
})

st.line_chart(prediction_df, x='Year', y='GDP', color='Country Code')
