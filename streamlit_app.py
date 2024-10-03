import streamlit as st
import pandas as pd
import numpy as np
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

# Memuat data GDP
gdp_df = get_gdp_data()

# Judul aplikasi
st.title('GDP Prediction Dashboard :earth_americas:')

# Filter berdasarkan negara dan tahun
countries = gdp_df['Country Name'].unique()
selected_countries = st.multiselect('Select countries:', countries, ['United States', 'China'])

min_year = gdp_df['Year'].min()
max_year = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Select the range of years:',
    min_value=min_year,
    max_value=max_year,
    value=[min_year, max_year]
)

# Filter data berdasarkan input pengguna
filtered_gdp_df = gdp_df[
    (gdp_df['Country Name'].isin(selected_countries)) &
    (gdp_df['Year'] >= from_year) &
    (gdp_df['Year'] <= to_year)
]

st.subheader('GDP Data:')
st.write(filtered_gdp_df)

# Model prediksi GDP dengan Polynomial Regression
@st.cache_data
def train_gdp_model(gdp_df, degree=3):
    """Melatih model prediksi GDP menggunakan Polynomial Regression."""
    X = gdp_df[['Year']].values
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
    model, mse, poly = train_gdp_model(filtered_gdp_df)

    # Pilih tahun untuk prediksi di masa depan
    future_years = st.multiselect(
        'Select future years to predict GDP:',
        options=[year for year in range(max_year + 1, 2051)],
        default=[2025, 2030, 2040]
    )

    if len(future_years) > 0:
        # Prediksi GDP untuk tahun-tahun yang dipilih
        future_years_arr = np.array(future_years).reshape(-1, 1)
        future_years_poly = poly.transform(future_years_arr)
        future_gdp_pred = model.predict(future_years_poly)

        st.subheader('GDP Predictions:')
        predictions_df = pd.DataFrame({
            'Year': future_years,
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
        st.warning('Please select future years to predict.')
else:
    st.warning('Please select countries and years to view data.')
