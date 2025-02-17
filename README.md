# Streamlit app to forecast short-term Photovoltaic energy production for domestic installations

This is a Streamlit app to forecast photovoltaic production using an LSTM model. This is just a kind of proof of concept. 
You can find the deployed app here: https://photovoltaic-forecast.streamlit.app/

The forecasts are valid for a particular solar installation. If you own a domestic off-grid installation which relies on batteries as 
backup energy source, you should be able to forecast next day energy production using this app so you can optimize your backup energy
planning.

Let's imagine you just got your new solar installation. In this app, you then can enter the location and other parameters of your
system on the left sidebar:
![creating_new_model](https://github.com/user-attachments/assets/e141d88e-82a2-4e08-bd81-fc887c9de9fb)

Once you enter the parameters, a new forecasting model will be created after clicking the "Fetch data and Create model" button.
Currently, the model is trained on historical data from PVGIS corresponding to the specified parameters. The model fetches data
from last two years.

After successful model creation, the app will provide an option to upload a CSV file with production data from the last 24 hours.
Using this uploaded file, the model will forecast production for the next 24 hours (the figures are not showing the actual
power values, but the scaled values. Currently working on that):
![data_preview_and_predictions](https://github.com/user-attachments/assets/0dd51e2b-fb76-47e8-8977-7f9c7cb9b32f)

Below the data preview and predictions tables, there's an additional section where you can find plots to visualize the forecast:
![forecast_vis](https://github.com/user-attachments/assets/8d5e96c9-a8ec-4f2f-94aa-0b5c6c59ffd5)

Any comments and suggestions are welcome !
