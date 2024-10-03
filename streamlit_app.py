import streamlit as st
import pandas as pd
import numpy as np
import requests  # Library untuk HTTP request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set title dan ikon
st.set_page_config(
    page_title='GDP Prediction Dashboard',
    page_icon=':earth_americas:',
)

# Fungsi untuk memuat data GDP
@st.cache_data
def get_gdp_data():
    """Memuat data GDP dari file CSV."""
    DATA_FILENAME = 'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)
    
    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # Ubah bentuk data agar lebih rapi
    gdp_df = raw_gdp_df.melt(
        id_vars=['Country Name', 'Country Code'],
        value_vars=[str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        var_name='Year',
        value_name='GDP'
    )
    
    # Ubah kolom tahun menjadi integer
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])
    
    # Hilangkan nilai GDP yang kosong
    gdp_df = gdp_df.dropna(subset=['GDP'])
    
    return gdp_df

# Fungsi untuk mengambil data suku bunga dari API SatuData
@st.cache_data
def get_interest_rate_data():
    """Mengambil data suku bunga dari SatuData API."""
    api_url = 'https://katalog.satudata.go.id/api/3/action/datastore_search'
    params = {
        'resource_id': '<RESOURCE_ID>',  # Ganti dengan resource ID untuk data suku bunga
        'limit': 1000
    }
    
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        records = data['result']['records']
        interest_rate_df = pd.DataFrame(records)
        
        # Konversi tahun dan suku bunga ke tipe numerik
        interest_rate_df['Year'] = pd.to_numeric(interest_rate_df['Year'])
        interest_rate_df['Interest Rate'] = pd.to_numeric(interest_rate_df['Interest Rate'])
        
        return interest_rate_df[['Year', 'Interest Rate']]
    else:
        st.error("Failed to fetch interest rate data from SatuData API.")
        return pd.DataFrame()

# Memuat data GDP
gdp_df = get_gdp_data()

# Memuat data suku bunga
interest_rate_df = get_interest_rate_data()

# Gabungkan data GDP dan suku bunga berdasarkan tahun
merged_df = pd.merge(gdp_df, interest_rate_df, on='Year', how='left')

# Judul aplikasi
st.title('GDP Prediction Dashboard :earth_americas:')

# Filter berdasarkan negara dan tahun
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

# Filter data berdasarkan input pengguna
filtered_gdp_df = merged_df[
    (merged_df['Country Name'].isin(selected_countries)) &
    (merged_gdp_df['Year'] >= from_year) &
    (filtered_gdp_df['Year'] <= to_year)
]

st.subheader('GDP Data with Interest Rates:')
st.write(filtered_gdp_df)

# Model prediksi GDP dengan Polynomial Regression, menggunakan Year dan Interest Rate
@st.cache_data
def train_gdp_model_with_interest_rate(gdp_df, degree=3):
    """Melatih model prediksi GDP menggunakan Polynomial Regression dengan faktor suku bunga."""
    X = gdp_df[['Year', 'Interest Rate']].values  # Tambahkan suku bunga sebagai input
    y = gdp_df['GDP'].values

    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Polynomial Features untuk meningkatkan akurasi
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Buat model regresi linear
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Prediksi pada data test
    y_pred = model.predict(X_test_poly)

    # Hitung kesalahan (Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse, poly

# Melatih model
if len(filtered_gdp_df) > 0:
    model, mse, poly = train_gdp_model_with_interest_rate(filtered_gdp_df)

    # Pilih tahun untuk prediksi di masa depan
    future_years = st.multiselect(
        'Select future years to predict GDP:',
        options=[year for year in range(max_year + 1, 2051)],
        default=[2025, 2030, 2040]
    )

    # Input suku bunga untuk tahun-tahun masa depan
    future_interest_rates = st.text_input('Enter future interest rates (comma-separated):', '3.5, 4.0, 4.5')

    if len(future_years) > 0 and future_interest_rates:
        # Prediksi GDP untuk tahun-tahun yang dipilih
        future_years_arr = np.array(future_years).reshape(-1, 1)
        future_interest_rates_arr = np.array([float(x) for x in future_interest_rates.split(',')]).reshape(-1, 1)
        future_data = np.hstack([future_years_arr, future_interest_rates_arr])

        future_data_poly = poly.transform(future_data)
        future_gdp_pred = model.predict(future_data_poly)

        st.subheader('GDP Predictions with Interest Rates:')
        predictions_df = pd.DataFrame({
            'Year': future_years,
            'Interest Rate': future_interest_rates_arr.flatten(),
            'Predicted GDP': future_gdp_pred
        })
        st.write(predictions_df)

        # Gabungkan data aktual dan prediksi
        actual_data = filtered_gdp_df[['Year', 'GDP']]
        predicted_data = predictions_df.rename(columns={'Predicted GDP': 'GDP'})
        combined_data = pd.concat([actual_data, predicted_data])

        # Plot hasil prediksi dan data aktual
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
        st.warning('Please select future years and enter interest rates to predict.')
else:
    st.warning('Please select countries and years to view data.')
