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
    page_title='GDP & Interest Rate Prediction Dashboard',
    page_icon=':earth_americas:',
)

# Function to load GDP and Interest Rate data
@st.cache_data
def get_data():
    """Load GDP and Interest Rate data from CSV files."""
    gdp_data_filename = 'data/gdp_data.csv'
    interest_rate_filename = 'data/API_FR.INR.RINR_DS2_en_csv_v2_5728810.csv'
    
    # Load GDP data
    raw_gdp_df = pd.read_csv(gdp_data_filename)
    
    # Load interest rate data
    raw_interest_rate_df = pd.read_csv(interest_rate_filename, skiprows=4)

    # Check column names in interest rate data
    st.write("Interest Rate Data Columns:", raw_interest_rate_df.columns.tolist())
    
    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # Reshape GDP data
    gdp_df = raw_gdp_df.melt(
        id_vars=['Country Name', 'Country Code'],
        value_vars=[str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        var_name='Year',
        value_name='GDP'
    )
    
    # Ensure columns exist before reshaping interest rate data
    if 'Country Name' in raw_interest_rate_df.columns and 'Country Code' in raw_interest_rate_df.columns:
        interest_rate_df = raw_interest_rate_df.melt(
            id_vars=['Country Name', 'Country Code'],
            value_vars=[str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
            var_name='Year',
            value_name='Interest Rate'
        )
    else:
        st.error("Expected columns 'Country Name' and 'Country Code' not found in interest rate data.")
        return None

    # Convert year column to numeric
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])
    interest_rate_df['Year'] = pd.to_numeric(interest_rate_df['Year'])
    
    # Drop rows with missing GDP and Interest Rate values
    gdp_df = gdp_df.dropna(subset=['GDP'])
    interest_rate_df = interest_rate_df.dropna(subset=['Interest Rate'])
    
    # Merge GDP and Interest Rate data by Country and Year
    merged_df = pd.merge(gdp_df, interest_rate_df, on=['Country Name', 'Country Code', 'Year'], how='inner')

    return merged_df

# Load GDP and Interest Rate data
data_df = get_data()

if data_df is not None:
    # Title of the app
    st.title('GDP & Interest Rate Prediction Dashboard :earth_americas:')

    # Filter by country and year
    countries = data_df['Country Name'].unique()
    selected_countries = st.multiselect('Select countries:', countries, ['United States', 'China'])

    min_year = data_df['Year'].min()
    max_year = data_df['Year'].max()

    from_year, to_year = st.slider(
        'Select the range of years:',
        min_value=min_year,
        max_value=max_year,
        value=[min_year, max_year]
    )

    # Filter data based on user input
    filtered_df = data_df[
        (data_df['Country Name'].isin(selected_countries)) &
        (data_df['Year'] >= from_year) &
        (data_df['Year'] <= to_year)
    ]

    st.subheader('GDP and Interest Rate Data:')
    st.write(filtered_df)

    # Model for predicting GDP with Interest Rate
    @st.cache_data
    def train_model(filtered_df, degree=3):
        """Train a polynomial regression model with Interest Rate as a feature."""
        X = filtered_df[['Year', 'Interest Rate']].values
        y = filtered_df['GDP'].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Polynomial Features to increase accuracy
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Create linear regression model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test_poly)

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)

        return model, mse, poly

    # Train the model
    if len(filtered_df) > 0:
        model, mse, poly = train_model(filtered_df)

        # Select future years for prediction
        future_years = st.multiselect(
            'Select future years to predict GDP:',
            options=[year for year in range(max_year + 1, 2051)],
            default=[2025, 2030, 2040]
        )

        if len(future_years) > 0:
            # Input Interest Rate for future years
            future_interest_rates = st.text_input(
                'Enter the future interest rates for the selected years (comma-separated):',
                '2.5, 3.0, 3.5'
            )

            # Convert interest rates to array
            future_interest_rates = [float(x.strip()) for x in future_interest_rates.split(',')]

            if len(future_interest_rates) == len(future_years):
                # Predict GDP for the selected future years
                future_data = np.column_stack([future_years, future_interest_rates])
                future_data_poly = poly.transform(future_data)
                future_gdp_pred = model.predict(future_data_poly)

                st.subheader('GDP Predictions:')
                predictions_df = pd.DataFrame({
                    'Year': future_years,
                    'Future Interest Rate': future_interest_rates,
                    'Predicted GDP': future_gdp_pred
                })
                st.write(predictions_df)

                # Combine actual and predicted data
                actual_data = filtered_df[['Year', 'GDP']]
                predicted_data = predictions_df.rename(columns={'Predicted GDP': 'GDP'})
                combined_data = pd.concat([actual_data, predicted_data])

                # Plot the actual and predicted data
                fig, ax = plt.subplots()
                ax.plot(actual_data['Year'], actual_data['GDP'], label='Actual GDP', marker='o', color='g')
                ax.plot(predicted_data['Year'], predicted_data['GDP'], label='Predicted GDP', marker='o', linestyle='--', color='b')
                ax.set_xlabel('Year')
                ax.set_ylabel('GDP')
                ax.set_title(f'GDP for {", ".join(selected_countries)} (Actual & Predicted)')
                ax.legend()
                st.pyplot(fig)

                st.write(f'Mean Squared Error of the model: {mse:.2f}')
            else:
                st.warning('Please ensure the number of interest rates matches the number of years selected.')
        else:
            st.warning('Please select future years to predict.')
else:
    st.warning('Failed to load data. Please check the dataset.')
