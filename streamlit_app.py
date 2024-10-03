import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set title and icon
st.set_page_config(
    page_title='GDP Prediction with Interest Rate Dashboard',
    page_icon=':earth_americas:',
)

# Function to load GDP data
@st.cache_data
def get_gdp_data():
    """Load GDP data from file."""
    gdp_file = 'data/gdp_data.csv'  # Replace with the path to your GDP file
    raw_gdp_df = pd.read_csv(gdp_file)
    
    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # Reshape the data
    gdp_df = raw_gdp_df.melt(
        id_vars=['Country Name', 'Country Code'],
        value_vars=[str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        var_name='Year',
        value_name='GDP'
    )
    
    # Convert year to numeric
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])
    
    # Drop missing GDP values
    gdp_df = gdp_df.dropna(subset=['GDP'])
    
    return gdp_df

# Function to load interest rate data
# Function to load interest rate data
@st.cache_data
def get_interest_rate_data():
    """Load interest rate data from file."""
    interest_rate_file = 'data/API_FR.INR.RINR_DS2_en_csv_v2_5728810.csv'  # Replace with the path to your interest rate file
    raw_interest_rate_df = pd.read_csv(interest_rate_file, skiprows=4)  # Adjust skiprows based on file structure
    
    # Display the first few rows of the DataFrame for inspection
    st.write("Interest Rate Data Preview:", raw_interest_rate_df.head())
    st.write("Interest Rate Data Columns:", raw_interest_rate_df.columns)

    # Define the range of years
    MIN_YEAR = 1960
    MAX_YEAR = 2022
    
    # Check if 'Country Name' and the year columns exist
    # Constructing the list of years dynamically based on the available columns
    year_columns = [str(year) for year in range(MIN_YEAR, MAX_YEAR + 1)]
    available_years = [col for col in raw_interest_rate_df.columns if col in year_columns]
    
    if 'Country Name' not in raw_interest_rate_df.columns:
        st.error("'Country Name' column not found in the dataset.")
        st.stop()
    
    if not available_years:
        st.error(f"No year columns found in the range {MIN_YEAR}-{MAX_YEAR}.")
        st.stop()
    
    # Now perform the melt operation
    interest_rate_df = raw_interest_rate_df.melt(
        id_vars=['Country Name'],  # Adjust based on the actual column names
        value_vars=available_years,
        var_name='Year',
        value_name='Interest Rate'
    )

    # Convert year to numeric
    interest_rate_df['Year'] = pd.to_numeric(interest_rate_df['Year'])
    
    # Drop missing interest rate values
    interest_rate_df = interest_rate_df.dropna(subset=['Interest Rate'])
    
    return interest_rate_df

# Load GDP and interest rate data
gdp_df = get_gdp_data()
interest_rate_df = get_interest_rate_data()

# Merge GDP and interest rate data on Country and Year
merged_df = pd.merge(gdp_df, interest_rate_df, how='inner', on=['Country Name', 'Year'])

# Application title
st.title('GDP Prediction with Interest Rate Dashboard :earth_americas:')

# Filter by country and year
countries = merged_df['Country Name'].unique()
selected_countries = st.multiselect('Select countries:', countries, ['United States', 'China'])

min_year = merged_df['Year'].min()
max_year = merged_df['Year'].max()

from_year, to_year = st.slider(
    'Select the range of years:',
    min_value=min_year,
    max_value=max_year,
    value=[min_year, max_year]
)

# Filter data based on user input
filtered_df = merged_df[
    (merged_df['Country Name'].isin(selected_countries)) &
    (merged_df['Year'] >= from_year) &
    (merged_df['Year'] <= to_year)
]

st.subheader('GDP and Interest Rate Data:')
st.write(filtered_df)

# Function to train GDP prediction model
@st.cache_data
def train_gdp_model(merged_df, degree=3):
    """Train a polynomial regression model for GDP prediction using interest rate."""
    X = merged_df[['Year', 'Interest Rate']].values  # Include Interest Rate in the model
    y = merged_df['GDP'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Polynomial features for higher accuracy
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Linear regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_poly)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return model, mse, poly

# Train the model
if len(filtered_df) > 0:
    model, mse, poly = train_gdp_model(filtered_df)

    # Select future years for GDP prediction
    future_years = st.multiselect(
        'Select future years to predict GDP:',
        options=[year for year in range(max_year + 1, 2051)],
        default=[2025, 2030, 2040]
    )

    if len(future_years) > 0:
        # Add Interest Rate for future prediction
        future_interest_rate = st.number_input('Enter Interest Rate for future years:', value=2.0)

        # Create future data for prediction
        future_data = pd.DataFrame({
            'Year': future_years,
            'Interest Rate': [future_interest_rate] * len(future_years)
        })

        # Predict GDP for future years
        future_years_arr = future_data[['Year', 'Interest Rate']].values
        future_years_poly = poly.transform(future_years_arr)
        future_gdp_pred = model.predict(future_years_poly)

        # Display predictions
        st.subheader('GDP Predictions for Future Years:')
        predictions_df = pd.DataFrame({
            'Year': future_years,
            'Predicted GDP': future_gdp_pred
        })
        st.write(predictions_df)

        # Plot actual vs predicted GDP
        actual_data = filtered_df[['Year', 'GDP']]
        predicted_data = predictions_df.rename(columns={'Predicted GDP': 'GDP'})
        combined_data = pd.concat([actual_data, predicted_data])

        # Plot
        fig, ax = plt.subplots()
        ax.plot(actual_data['Year'], actual_data['GDP'], label='Actual GDP', marker='o', color='g')
        ax.plot(predicted_data['Year'], predicted_data['GDP'], label='Predicted GDP', marker='o', linestyle='--', color='b')
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.set_title(f'GDP Prediction for {", ".join(selected_countries)}')
        ax.legend()
        st.pyplot(fig)

        st.write(f'Mean Squared Error of the model: {mse:.2f}')
    else:
        st.warning('Please select future years to predict.')
else:
    st.warning('Please select countries and years to view data.')
