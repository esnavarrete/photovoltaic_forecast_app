import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

import plotly.express as px
import plotly.graph_objects as go

from sklearn import set_config
from keras.models import load_model
from modelling import PVGIS_Modelling

set_config(transform_output='pandas')


def fetch_pvgis_data(latitude, longitude, peak_power, tech, loss, mount):
    base_url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    
    # Rango de tiempo de los datos históricos (últimos 2 años)
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=730)
    
    params = {
        'lat': latitude,
        'lon': longitude,
        'startyear': 2020,
        'endyear': 2023,
        'pvcalculation': 1,
        'peakpower': peak_power,
        'loss': loss,
        'mountingplace': mount,
        'pvtechchoice': tech,
        'outputformat': 'json',
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while extracting data from PVGIS: {str(e)}")
        return None


def make_prediction(model, df_to_predict):
    y_preds = model.predict(df_to_predict['scaled_P'])
    y_preds = pd.DataFrame(y_preds, columns=['future_preds'])

    # Se muestra el resultado
    final_result = pd.concat([df_to_predict, y_preds], axis=1)
    return final_result


def create_visualizations(df):
    future_values = df[['future_preds']].copy()
    future_values['time'] = df['time'] + pd.Timedelta(hours=24)
    future_values['type'] = 'Future values'
    future_values = future_values.rename(columns={'future_preds': 'scaled_P'})

    df_both = df.drop(columns=['future_preds']).copy()
    df_both['type'] = 'Present values'
    df_both = pd.concat([df_both, future_values], axis=0)

    # Present and future values plot
    present_future_fig = px.line(df_both,
                                 x='time', 
                                 y='scaled_P', 
                                 title='Present and future values for PV power production',
                                 color='type',
                                 labels={'time': 'Time hour by hour', 'scaled_P': 'Scaled Power'})

    # Only future values plot
    only_future_fig = px.line(future_values, 
                              x='time', 
                              y='scaled_P', 
                              title='Forecasted PV production values for next day',
                              labels={"time": "Time hour by hour", "scaled_P": "Scaled Power"})

    return present_future_fig, only_future_fig



# Streamlit app
def main():
    st.title("Photovoltaic Production Forecast Tool")
    
    # DEFINING PARAMETERS FOR A NEW FORECAST MODEL
    st.sidebar.header("Create new forecast model")
    
    # PV system location
    latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=48.8566)
    longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=2.3522)
    
    # Other parameters
    peak_power = st.sidebar.number_input("Maximum Power (kW)", min_value=0.1, value=1.0)
    
    # PV system technology type
    technology = st.sidebar.selectbox(
        "Photovoltaic Technology",
        ["crystSi", "CIS", "CdTe", "Unknown"],
        index=0
    )

    # PV systems' natural loss
    loss = st.sidebar.slider("System Loss (%)", min_value=0, max_value=100, value=14)
    
    # PV system mounting type
    mounting = st.sidebar.selectbox(
        "Mounting Type",
        ["free", "building"],
        index=0
    )

    # forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=7, value=3)

    if st.sidebar.button("Fetch data and Create model"):
        try:
            with st.spinner("Fetching historical data ..."):
                # data extraction
                data = fetch_pvgis_data(latitude, longitude, peak_power, technology, loss, mounting)
            if len(data.keys()) != 3:
                raise
        except Exception as e:
            st.error(f'An error occurred while fetching data. Try again: {e}')
        else:  
            st.success('Data fetched successfully!')  
            with st.spinner("Creating forecast model ..."):
                df = pd.DataFrame(data['outputs']['hourly'])
                df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')

                modeler = PVGIS_Modelling()
                modeler.model_generation(df)

    if 'forecast_clicked' not in st.session_state:
            st.session_state.forecast_clicked = False

    if st.button('Forecast production with existing model'):
        st.session_state.forecast_clicked = True
        
    if os.path.exists('model.keras') and st.session_state.forecast_clicked:
        loaded_model = load_model('model.keras')

        if 'predict_clicked' not in st.session_state:
            st.session_state.predict_clicked = False

        uploaded_file = st.file_uploader("Upload CSV file for forecasting", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df['time'] = pd.to_datetime(df['time'])
            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader("Recent Data Preview:")
                st.dataframe(df, use_container_width=True)
                if st.button("Make Prediction"):
                    st.session_state.predict_clicked = True

            if st.session_state.predict_clicked:
                predictions = make_prediction(loaded_model, df)
                with right_col:                
                    # Show predictions if button was clicked
                    st.subheader("Predictions for next day:")
                    st.dataframe(predictions, use_container_width=True)
                
                st.subheader('Forecast Visualization')
                tab1, tab2 = st.tabs(['Present and future values', 'Only future values'])
                fig1, fig2 = create_visualizations(predictions)

                with tab1:
                    st.plotly_chart(fig1, use_container_width=True)
                    st.write("""
                        This plot shows current actual values and the forecasted values for 
                        the next day.
                    """)
                with tab2:
                    st.plotly_chart(fig2, use_container_width=True)
                    st.write("""
                        This plot shows only the forecasted values for the next day.
                    """)

                # Add option to reset
                if st.button("Reset"):
                    st.session_state.forecast_clicked = False
                    st.session_state.predict_clicked = False
                    st.rerun()
                      
    elif not os.path.exists('model.keras'):
            st.error("No forecast model was detected. Try creating a new one.")

if __name__ == "__main__":
    main()