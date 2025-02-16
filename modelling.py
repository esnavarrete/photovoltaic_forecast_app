import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn import set_config
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from keras import metrics
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam

set_config(transform_output='pandas')

class PVGIS_Modelling:

    def create_lstm_model(self, sequence_length=24):
        model = Sequential([
                LSTM(units=128,  
                    input_shape=(sequence_length, 1),
                    activation='tanh'),
                Dense(units=1)
            ])
        
        early_stopping = EarlyStopping(
                monitor='val_loss',
                #patience=5,
                restore_best_weights=True
            )
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mse', 'mean_absolute_percentage_error'])
        
        return model, early_stopping


    def prepare_data_for_lstm(self, data):
        df_train = data[['time', 'P']]

        # Escalamiento de los datos
        scaler = StandardScaler()
        df_train['scaled_P'] = scaler.fit_transform(df_train[['P']])

        # Definición de la variable objetivo
        df_train['scaled_P_t+24'] = df_train['scaled_P'].shift(-24)
        df_train = df_train[:-24]
        return df_train


    def model_generation(self, df):
        
        df_train = self.prepare_data_for_lstm(df)

        # Creación y entrenamiento del modelo
        model, early_stopping = self.create_lstm_model(24)
        model.fit(df_train['scaled_P'], df_train['scaled_P_t+24'],
                epochs=100, 
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1)
        
        model.save("model.keras")
        return


if __name__ == "__main__":
    # testing code

    df = pd.read_csv('datos_de_prueba.csv')
    print(df.shape)
    scaler = StandardScaler()
    df['scaled_P'] = scaler.fit_transform(df[['P']])
    print(df)