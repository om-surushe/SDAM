import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np


def predict_crypto():
    # Load the dataset
    data = pd.read_csv('crypto_price.csv')

    # Set the initial number of days to use for the moving average
    window_size = 10

    # Calculate the moving average
    data['MA'] = data['Price'].rolling(window_size).mean()

    # Extrapolate the next 31 days of prices based on the moving average and the average change
    predictions = []
    for i in range(31):
        # Get the most recent window_size days of data
        last_window = data.tail(window_size)

        # Calculate the average change in price over the last window_size days
        average_change = np.mean(np.diff(last_window['Price']))

        # Extrapolate the next day's price based on the last moving average and the average change
        if i == 0:
            prediction = data['Price'].iloc[-1]
        else:
            # Update the window size based on the number of predictions made so far
            window_size = min(i*2, len(data)-1)

            # Calculate the new moving average and make the prediction
            data['MA'] = data['Price'].rolling(window_size).mean()
            prediction = data['MA'].iloc[-1] + average_change

        # Append the prediction to the list of predictions
        predictions.append(prediction)

        # Update the dataset with the new prediction
        new_date = pd.date_range(data['Date'].iloc[-1], periods=2, freq='D')[1]
        new_data = pd.DataFrame(
            {'Date': new_date, 'Price': prediction}, index=[data.index[-1]+1])
        data = pd.concat([data, new_data], ignore_index=False)

    return predictions[:30]

# Creating a function to train the model


def get_model():
    df = pd.read_csv('gold_price.csv')

    # Convert the date column to datetime type and set it as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Scaling the price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

    # Creating a function to create the LSTM model
    def create_lstm_model():
        model = Sequential()
        model.add(LSTM(units=80, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(units=80))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Creating the training data
    x_train, y_train = [], []
    for i in range(60, scaled_price.shape[0]):
        x_train.append(scaled_price[i-60:i, 0])
        y_train.append(scaled_price[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshaping the data for the LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Creating the LSTM model
    model = create_lstm_model()

    # Training the LSTM model
    model.fit(x_train, y_train, epochs=10, batch_size=25)

    # Creating the testing data
    test_data = scaled_price[-60:]
    x_test = []
    x_test.append(test_data)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Saving the model
    model.save('gold_model.h5')

# Creating a function to load the model and predict the price


def predict_gold():
    # Checking if the model is already trained
    try:
        model = load_model('gold_model.h5')
    except:
        get_model('gold_price.csv')
        model = load_model('gold_model.h5')

    df = pd.read_csv('gold_price.csv')

    # Convert the date column to datetime type and set it as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Scaling the price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

    # Creating the testing data
    test_data = scaled_price[-60:]
    x_test = []
    x_test.append(test_data)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predicting the prices using the LSTM model for the next 30 days
    predicted_prices = []
    for i in range(30):
        predicted_price = model.predict(x_test)
        predicted_prices.append(predicted_price[0, 0])
        x_test = np.concatenate(
            (x_test[:, 1:, :], predicted_price.reshape(1, 1, 1)), axis=1)

    # Scaling the predicted prices back to their original range
    predicted_prices = scaler.inverse_transform(
        np.array(predicted_prices).reshape(-1, 1))

    # Converting the predicted prices to a list
    predicted_prices = predicted_prices.reshape(1, -1)[0].tolist()
    return predicted_prices


def predict_real_estate():
    real_estate_df = pd.read_csv("real_estate_price.csv")

    # Prepare data
    real_estate_df['Date'] = pd.to_datetime(real_estate_df['Date'])
    real_estate_df.set_index('Date', inplace=True)
    real_estate_df = real_estate_df.resample('M').interpolate(method='linear')

    # Train models
    real_estate_model = sm.tsa.ARIMA(real_estate_df, order=(1, 1, 0))
    real_estate_result = real_estate_model.fit()

    # Generate forecasts
    real_estate_forecast = real_estate_result.predict(
        start=len(real_estate_df), end=len(real_estate_df)+29, typ='levels')

    return np.array(real_estate_forecast)


def predict_stocks():

    # Load the dataset
    data = pd.read_csv('stock_price.csv')
    window_size = 10

    # Calculate the moving average
    data['MA'] = data['Price'].rolling(window_size).mean()

    # Extrapolate the next 31 days of prices based on the moving average and the average change
    predictions = []
    for i in range(31):
        # Get the most recent window_size days of data
        last_window = data.tail(window_size)

        # Calculate the average change in price over the last window_size days
        average_change = np.mean(np.diff(last_window['Price']))

        # Extrapolate the next day's price based on the last moving average and the average change
        if i == 0:
            prediction = data['Price'].iloc[-1]
        else:
            # Update the window size based on the number of predictions made so far
            window_size = min(i*2, len(data)-1)

            # Calculate the new moving average and make the prediction
            data['MA'] = data['Price'].rolling(window_size).mean()
            prediction = data['MA'].iloc[-1] + average_change

        # Append the prediction to the list of predictions
        predictions.append(prediction)

        # Update the dataset with the new prediction
        new_date = pd.date_range(data['Date'].iloc[-1], periods=2, freq='D')[1]
        new_data = pd.DataFrame(
            {'Date': new_date, 'Price': prediction}, index=[data.index[-1]+1])
        data = pd.concat([data, new_data], ignore_index=False)

    # Return the predictions for the next 30 days
    return predictions[1:]


crypto = predict_crypto()
gold = predict_gold()
real_estate = predict_real_estate()
stocks = predict_stocks()

# Normalize the predictions from each model in a range from 0 to 100
crypto = (crypto - np.min(crypto)) / (np.max(crypto) - np.min(crypto)) * 100
gold = (gold - np.min(gold)) / (np.max(gold) - np.min(gold)) * 100
real_estate = (real_estate - np.min(real_estate)) / \
    (np.max(real_estate) - np.min(real_estate)) * 100
stocks = (stocks - np.min(stocks)) / (np.max(stocks) - np.min(stocks)) * 100

# Create a dataframe with the all the predictions combined
predictions = pd.DataFrame(
    {'Crypto': crypto, 'Gold': gold, 'Real Estate': real_estate, 'Stocks': stocks})


st.line_chart(predictions)
