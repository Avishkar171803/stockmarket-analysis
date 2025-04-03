import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load trained model
model = load_model(r'D:\Lectures\stock market prediction\stock_model.keras')

# Streamlit UI
st.header("Stock Market Prediction")

# Get stock symbol input
stock = st.text_input("Enter symbol of stock", "AAPL")
start = "2023-01-01"
end = "2024-01-01"

# Fetch stock data
st.subheader("Stock Data")
st.write("Fetching data for", stock)

data = yf.download(stock, start=start, end=end)

# Check if 'Close' column exists
if 'Close' not in data.columns:
    st.error("The 'Close' column is missing from the dataset.")
    st.stop()

# Convert 'Close' column to DataFrame and split into train & test sets
data_train = pd.DataFrame(data['Close']).iloc[:int(len(data) * 0.8)]
data_test = pd.DataFrame(data['Close']).iloc[int(len(data) * 0.8):]

# Scale the test data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)  # Fit scaler on training data

last_100_days = data_train.tail(100)
data_combined = pd.concat([last_100_days, data_test], ignore_index=True)

data_test_scaled = scaler.transform(data_combined)

# Calculate 100-day moving average
st.subheader("100-Day Moving Average")
ma_100_days = data['Close'].rolling(window=100, min_periods=1).mean()  # Ensure early periods are not NaN
st.write(ma_100_days)

# Plot the closing price and moving average
fig = plt.figure(figsize=(10, 8))
plt.plot(data.index, data['Close'], 'r', label="AAPL Closing Price")
plt.plot(data.index, ma_100_days, 'g', label="100-Day Moving Average", linewidth=2)
plt.legend()
plt.savefig('ma_100_days.png')
st.pyplot(fig)

# Prepare input sequences
x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Display additional information
st.subheader("Raw Stock Data")
st.write(data)  # Display fetched stock data

st.subheader("Training Data Preview")
st.write(data_train.head())  # Show first few rows of training data

st.subheader("Testing Data Preview")
st.write(data_test.head())  # Show first few rows of testing data

st.subheader("Scaled Data Shape")
st.write(f"x shape: {x.shape}, y shape: {y.shape}")  # Display input data shape

st.success("Data processing complete!")
