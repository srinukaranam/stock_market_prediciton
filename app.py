import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Fetch real-time stock data
st.title("📈 Real-Time Stock Price Prediction")
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")  # Default: Apple (AAPL)
data = yf.download(ticker, period="5y", interval="1d")  # Fetch last 5 years of data

if data.empty or len(data) < 100:
    st.error("⚠️ Not enough data available. Try another stock or wait for more data.")
    st.stop()

data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Display company information
st.sidebar.header("Company Info")
company_info = yf.Ticker(ticker).info
st.sidebar.write(f"**Company Name:** {company_info.get('longName', 'N/A')}")
st.sidebar.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
st.sidebar.write(f"**Exchange:** {company_info.get('exchange', 'N/A')}")

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Open', 'Close']])

# Prepare training data
sequence_length = 60
x_train, y_train = [], []

for i in range(sequence_length, len(scaled_data) - 7):  
    x_train.append(scaled_data[i-sequence_length:i])  
    y_train.append(scaled_data[i, :2])  

x_train, y_train = np.array(x_train), np.array(y_train)

# ✅ Check if x_train and y_train are empty before training
if len(x_train) == 0 or len(y_train) == 0:
    st.error("⚠️ Not enough training data. Try selecting another stock.")
    st.stop()

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(sequence_length, 2)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(2)  # Predict Open & Close price
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (silent training)
model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0)

# ✅ Fixed Predict Function
def predict_price(input_date):
    input_date = pd.to_datetime(input_date)

    if input_date in data.index:
        input_index = data.index.get_loc(input_date)
    else:
        input_index = len(data)  # If future date, use next available index

    if input_index < sequence_length:
        return None, None  # Fixes "Not enough data" error

    input_data = scaled_data[input_index-sequence_length:input_index]
    input_data = np.expand_dims(input_data, axis=0)

    prediction = model.predict(input_data)[0]

    # ✅ Ensure inverse_transform works properly
    if not hasattr(scaler, "data_min_"):
        return None, None  # Prevents inverse_transform error

    dummy_array = np.zeros((1, scaled_data.shape[1]))  # Match dataset shape
    dummy_array[0, 0] = prediction[0]  # Open Price
    dummy_array[0, 1] = prediction[1]  # Close Price

    prediction_actual = scaler.inverse_transform(dummy_array)[0]

    return float(prediction_actual[0]), float(prediction_actual[1])  # Return Open & Close prices

# Date input for prediction
selected_date = st.date_input("📅 Select a Date to Predict Stock Price", datetime.today() + timedelta(days=1))

if st.button("Predict"):
    open_price, close_price = predict_price(selected_date)

    if open_price is None:
        st.error("⚠️ Prediction unavailable. Try a different date.")
    else:
        st.success(f"📌 Predicted Open Price: **${open_price:.2f}**")
        st.success(f"📌 Predicted Close Price: **${close_price:.2f}**")

# Future 7-day stock price graph
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 8)]
future_predictions = []

for date in future_dates:
    open_price, close_price = predict_price(date)
    if open_price is not None and close_price is not None:
        future_predictions.append((open_price, close_price))

future_df = pd.DataFrame(future_predictions, columns=['Open', 'Close'], index=future_dates)

st.subheader("📊 Future Stock Price Trend (Next 7 Days)")
fig, ax = plt.subplots()
ax.plot(future_df.index, future_df['Open'], marker='o', label="Predicted Open Price", color='blue')
ax.plot(future_df.index, future_df['Close'], marker='x', label="Predicted Close Price", color='red')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)
