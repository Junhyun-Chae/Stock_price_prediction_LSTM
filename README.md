# 📈 Stock Price Prediction GUI using Bidirectional LSTM

This project is a **graduation research application** that builds a desktop GUI to predict future stock prices using a **deep learning-based Bidirectional LSTM model**, and visualizes the results in an intuitive and user-friendly format.  
It is developed for **research and educational purposes only**, and **not intended for actual investment decisions**.

---

## 🔍 Project Overview

### 📌 Purpose
The goal of this project is to utilize a **Long Short-Term Memory (LSTM)** model to predict future stock prices based on historical data and present the results through a GUI program that enables users to easily select stocks and view the forecast with visual graphs.

---

## 🛠️ Methodology

### 1. Data Collection & Preprocessing
- **Input Data**: Stock price CSV files from Yahoo Finance (mainly using `Date` and `Close` columns)
- **Normalization**: Stock prices are scaled to the range 0–1 using `MinMaxScaler`
- **Sequence Creation**: Each input sequence consists of the past 60 business days of prices
- **Data Splitting**: Chronologically split into training (60%), validation (20%), and test sets (20%)

### 2. Model Architecture
- **Three layers of Bidirectional LSTM** are used to capture forward and backward dependencies
- **Dropout (0.2)** is applied between layers to prevent overfitting
- The output layer is a **Dense(1)** node for single-value prediction (closing price)
- **EarlyStopping** is applied when validation loss stops improving

### 3. Prediction & Evaluation
- The model is evaluated on the test set and predictions are inverse-transformed back to actual price scale
- Visual comparison between actual vs. predicted prices using `matplotlib` and `Plotly`
- Additionally, the model predicts the **next 60 days** sequentially for long-term forecasting

### 4. GUI Implementation
- A GUI is built using `tkinter`, allowing the user to select a stock (KOSPI or NASDAQ)
- Upon selection, the model automatically trains and displays prediction results graphically

---

## 📈 Results

- The trained LSTM model successfully tracks trends in the actual test data
- Visual comparison between actual and predicted prices shows reasonable accuracy
- Future price predictions for the next 60 business days are generated and visualized

> 🔸 For some stocks, precomputed `.xlsx` files are loaded and displayed for faster visualization

---

## 💡 Implications & Future Extensions

- LSTM-based forecasting shows potential in capturing stock price trends
- The GUI enables non-experts to interact with AI-driven forecasting tools easily

### Possible future improvements:
- Add technical indicators (e.g., RSI, MACD)
- Incorporate multiple input variables (e.g., volume, news sentiment)
- Apply newer architectures such as Transformer
- Evaluate using quantitative metrics (e.g., MAE, RMSE)

---

## ⚠️ Disclaimer

This project was developed as a **graduation research project** for academic purposes only.  
It is **not intended for real-world investment use**, and **no responsibility is taken for any financial decisions** made based on this code or results.

---

## 🖥️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- matplotlib
- plotly
- tkinter

---

## 🏃‍♂️ How to Run

1. Place stock CSV files into the `./data/` directory  
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
