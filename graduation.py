import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
import tkinter.messagebox as msgbox
import os
import pandas as pd
import plotly.graph_objects as go

current_dir = "./data"

csv_files = ['AAPL.csv', 'AMZN.csv', 'ASML.csv', 'AVGO.csv', 'COST.csv', 'GOOGL.csv', 'Hyundai_Motor_Company.csv', 'KAKAO.csv', 'Kia.csv', 'LG_Chem.csv', 'MSFT.csv', 'NAVER.csv', 'NVDA.csv', 'PEP.csv', 'POSCO.csv', 'Samsung_Biologics.csv', 'Samsung_Electronics_co.csv', 'Samsung_Electronics.csv', 'Samsung_SDI.csv', 'SK_hynix_Inc.csv', 'TSLA.csv']
dataframes = []

for csv_file in csv_files:
    file_path = os.path.abspath(os.path.join(current_dir,csv_file))
    df = pd.read_csv(file_path)
    dataframes.append(df)

########################################################
# set the title of the window

window = tk.Tk()
window.title("주가 예측 프로그램")

# create a frame for the title
title_frame = tk.Frame(window, width=400, height=50)
title_frame.pack(side=tk.TOP)

# create a label for the title
title_label = tk.Label(title_frame, text="예측하고 싶은 주식종목을 선택해주세요")
title_label.pack()

# divide the screen in half
left_frame = tk.Frame(window, width=200, height=400)
left_frame.pack(side=tk.LEFT)
right_frame = tk.Frame(window, width=200, height=400)
right_frame.pack(side=tk.RIGHT)

# add a title to the left frame
left_label = tk.Label(left_frame, text="KOSPI")
left_label.pack()

# add radio buttons to the left frame
var = tk.StringVar()
var.set("Option 1")
left_options = ["삼성전자", "SK하이닉스", "삼성바이오로직스", "삼성SDI", "LG화학"]
left_codes = ["Samsung_Electronics.csv", "SK_hynix_Inc.csv", "Samsung_Biologics.csv", "Samsung_SDI.csv", "LG_Chem.csv"]
left_dict = {option: code for option, code in zip(left_options, left_codes)}

for option in left_options:
    left_rb = tk.Radiobutton(left_frame, text=option, variable=var, value=option)
    left_rb.pack(anchor=tk.W)

# add a title to the right frame
right_label = tk.Label(right_frame, text="NASDAQ")
right_label.pack()

# add radio buttons to the right frame
right_options = ["애플", "마이크로소프트", "알파벳 A", "아마존닷컴", "엔비디아"]
right_codes = ["AAPL.csv", "MSFT.csv", "GOOGL.csv", "AMZN.csv", "NVDA.csv"]
right_dict = {option: code for option, code in zip(right_options, right_codes)}

for option in right_options:
    right_rb = tk.Radiobutton(right_frame, text=option, variable=var, value=option)
    right_rb.pack(anchor=tk.W)

# function to save the selected value
def save_value():
    global kabu
    kabu = left_dict.get(var.get()) or right_dict.get(var.get())
    msgbox.showinfo("주가 예측", "{}에 대해 예측중입니다.".format(kabu))
    window.destroy()
    return kabu

# add a button to save the selected value
save_button = tk.Button(window, text="Save", command=save_value)
save_button.pack()

# start the main event loop
window.mainloop()

# print the selected value
print("Selected value:", kabu)


##########################################################

# Load the data
df = pd.read_csv(kabu)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Preprocess and normalize the data
data = df[["Close"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Split the data into training, validation, and testing sets
train_data, test_data = train_test_split(data_scaled, test_size=0.2, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)

# Define the function to create input/output sequences
def create_dataset(data, time_steps):
    x_data, y_data = [], []
    for i in range(len(data) - time_steps - 1):
        x_data.append(data[i:(i+time_steps), :])
        y_data.append(data[i + time_steps, 0])
    return np.array(x_data), np.array(y_data)

# Create the input/output sequences
time_steps = 60
x_train, y_train = create_dataset(train_data, time_steps)
x_val, y_val = create_dataset(val_data, time_steps)
x_test, y_test = create_dataset(test_data, time_steps)

# Define and compile the LSTM model with 128 units and dropout
lstm_units = 128
dropout_rate = 0.2
model = Sequential()
model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True), input_shape=(x_train.shape[1], 1)))
model.add(Dropout(dropout_rate))
model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
model.add(Dropout(dropout_rate))
model.add(Bidirectional(LSTM(units=lstm_units)))
model.add(Dropout(dropout_rate))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model using the training and validation sets
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=83, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the test set and generate future predictions
predicted = model.predict(x_test)

# Inverse scale the predictions
predicted_prices = scaler.inverse_transform(predicted)

# Plot the actual and predicted data on the same graph
plt.figure(figsize=(16, 8))
plt.plot(df.index, df["Close"], label="Actual Data")
plt.plot(df.index[-len(predicted_prices):], predicted_prices, label="Predicted Data")
plt.xlabel("Date", fontsize=18)
plt.ylabel("Closing Price ($)", fontsize=18)
plt.title(f"Stock Prediction for {kabu}", fontsize=20)
plt.legend()
plt.show()

# Predict the next 60 days
next_60_days = []
last_sequence = data_scaled[-time_steps:]
for _ in range(60):
    next_sequence = model.predict(last_sequence.reshape(1, time_steps, 1))
    next_60_days.append(next_sequence[0, 0])
    last_sequence = np.concatenate((last_sequence[1:], next_sequence), axis=None)


if kabu==('AAPL.csv'):
    excel_data = pd.read_excel('AAPL.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('MSFT.csv'):
    excel_data = pd.read_excel('MSFT.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('GOOGL.csv'):
    excel_data = pd.read_excel('GOOGL.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('AMZN.csv'):
    excel_data = pd.read_excel('AMZN.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('NVDA.csv'):
    excel_data = pd.read_excel('NVDA.xlsx') 
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('Samsung_Electronics.csv'):
    excel_data = pd.read_excel('Samsung_Electronics.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('SK_hynix_Inc.csv'):
    excel_data = pd.read_excel('SK_hynix_Inc.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('Samsung_Biologics.csv'):
    excel_data = pd.read_excel('Samsung_Biologics.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('Samsung_SDI.csv'):
    excel_data = pd.read_excel('Samsung_SDI.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data
if kabu==('LG_Chem.csv'):
    excel_data = pd.read_excel('LG_Chem.xlsx')
    loaded_data = excel_data['Predicted_Price'].values.reshape(-1, 1)
    next_60_days_rescaled = loaded_data    

# Generate future dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=61, freq='B')[1:]

# Plot the actual and predicted data on the same graph
plt.figure(figsize=(16, 8))
plt.plot(df.index, df["Close"], label="Actual Data")
plt.plot(future_dates, next_60_days_rescaled, label="Predicted Data")
plt.xlabel("Date", fontsize=18) 
plt.ylabel("Closing Price ($)", fontsize=18)
plt.title(f"Stock Prediction for {kabu}", fontsize=20)
plt.legend()
plt.show()

# Plot the actual and predicted data on the same graph
fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Actual Data'))
fig.add_trace(go.Scatter(x=df.index[-len(predicted_prices):], y=predicted_prices.flatten(), mode='lines', name='Predicted Data'))

fig.update_layout(title=f"Stock Prediction for {kabu}",
                  yaxis_title="Closing Price ($)")
fig.update_xaxes(title="Date", tickformat="%b %d")

fig.show()

# Predict the next 60 days and plot them
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Actual Data'))
fig2.add_trace(go.Scatter(x=future_dates, y=next_60_days_rescaled.flatten(), mode='lines', name='Predicted Data'))

fig2.update_layout(title=f"Stock Prediction for {kabu} (Next 60 Days)",
                   yaxis_title="Closing Price ($)")
fig2.update_xaxes(title="Date", tickformat="%b %d")
fig2.show()