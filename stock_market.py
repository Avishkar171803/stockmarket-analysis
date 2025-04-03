import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch stock data (Example: Apple stock)
data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# Reset index and convert date
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate 100-day moving average
ma_100_days = data['Close'].rolling(window=100, min_periods=1).mean()  # Ensure early periods are not NaN
ma_200_days = data['Close'].rolling(window=200, min_periods=1).mean()  # Ensure early periods are not NaN
# Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], 'r', label="AAPL Closing Price")
plt.plot(data.index, ma_100_days, 'g', label="100-Day Moving Average", linewidth=2)
plt.plot(ma_200_days, 'y',  linewidth=2)
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title("AAPL Stock Price with 100-Day Moving Average")
plt.legend()
plt.grid()
plt.show()
data.dropna(inplace=True)
print(data.head())
data_train = pd.DataFrame(data['Close'])[0:int(len(data)*0.8)]
data_test = pd.DataFrame(data['Close'])[int(len(data)*0.8):int(len(data))]
data_train.shape[0], data_test.shape[0]
print(data_train.shape[0], data_test.shape[0])
data.shape[0]
print(data.shape[0])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)
# data_train_scaled.shape
# print(data_train_scaled.shape)
x=[]
y=[]
for i in range(100, data_train_scaled.shape[0]):
    x.append(data_train_scaled[i-100:i])
    y.append(data_train_scaled[i])  
# from keras.models import Dense, LSTM, Dropout
x, y = np.array(x), np.array(y)
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM, Dropout # type: ignore
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x[0].shape[1], 1)))
input_shape=(x[0].shape[1], 1)
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=50, batch_size=32,verbose=1)
model.summary()
pass_100_days = data_train.tail(100)
data_test=pd.concat((pass_100_days, data_test), ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)
x=[]
y=[]
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)
y_pred = model.predict(x)
print(y_pred)
# y_pred = scaler.inverse_transform(y_pred)
scale= 1/scaler.scale_
y_pred = y_pred*scale
y= y*scale
plt.figure(figsize=(14,5))
plt.plot(y, color = 'red', label = 'Real Apple Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Apple Stock Price')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.savefig("plot.png")
model.save("stock_model.keras")