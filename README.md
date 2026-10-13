# Stock_price_prediction_LSTM

Undergraduate graduation project (team of 2). A small tkinter app that predicts stock closing prices with a bidirectional LSTM.

You pick one of 10 stocks (KOSPI: Samsung Electronics, SK hynix, Samsung Biologics, Samsung SDI, LG Chem / NASDAQ: Apple, Microsoft, Alphabet, Amazon, NVIDIA), and the script trains the model on that stock's closing prices and plots the results.

## Model

- Daily closing prices from Yahoo Finance CSVs (`Date`, `Close`), scaled to 0–1 with `MinMaxScaler`
- 60-day input window; train / validation / test split 60 / 20 / 20 in time order
- 3 bidirectional LSTM layers (128 units) with dropout 0.2, then `Dense(1)`
- Adam, MSE loss, early stopping on validation loss (patience 10)

## Output

- Actual vs. predicted prices for the test period (matplotlib and Plotly)
- A 60-business-day forecast. To make it show up quickly, the forecast plot loads precomputed predictions from `<stock>.xlsx` (`Predicted_Price` column).

## Running

The CSV and xlsx files aren't in this repo. Put the CSVs in `./data/` and next to the script, then:

```bash
pip install pandas numpy matplotlib plotly scikit-learn tensorflow openpyxl
python graduation.py
```

School project, not investment advice.
