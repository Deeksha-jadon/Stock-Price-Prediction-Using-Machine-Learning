# Stock-Price-Prediction-Using-Machine-Learning
**Libraries used**

**Data Handling & Analysis**
(import pandas as pd
import numpy as np)

**Data Visualization**
(import matplotlib.pyplot as plt
import seaborn as sns)

**Stock Data Acquisition**
(import yfinance as yf)

**Machine Learning (Linear Regression)**
(from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score)

**Deep Learning (LSTM)**
(from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout)

**Date Handling**
(import datetime)

**Project Objective**
To predict the future stock prices of Infosys Ltd. (INFY.NS) using historical stock data by applying both traditional (Linear Regression) and deep learning (LSTM) techniques. The goal is to evaluate and compare the performance of these models on real-world financial data.

**Project Description**
This project uses secondary data collected from Yahoo Finance via the yfinance API. It applies both statistical and neural network models to forecast stock prices based on historical trends and patterns.

**Key Steps:**
**Data Acquisition**
Fetch historical daily stock data for Infosys from 2015 to present using the yfinance Python library.

**Data Preprocessing & Feature Engineering**
Extract and clean the 'Close' price.
Create lagged features like Prev_Close, SMA_10, SMA_30 for Linear Regression.
Normalize and reshape data for LSTM input.

**Exploratory Data Analysis**
Plot historical price charts.
Visualize moving averages.
Generate a correlation matrix to assess feature relationships.

**Linear Regression Modeling**
Use engineered features to predict the next day's closing price.
Evaluate model using metrics: MSE, RMSE, MAE, R².

**LSTM Neural Network Modeling**
Construct a multi-layered LSTM network to model time series behavior.
Train and evaluate the model on scaled stock data.
Plot actual vs predicted values to visualize model accuracy.

 **Future Price Prediction**
Predict stock prices for the next 30 days using the trained LSTM model.
Plot both historical and future prices.

**Key Findings**

Linear Regression provides a basic benchmark with interpretable results, but lacks temporal memory.

LSTM model performs better at capturing complex temporal patterns in stock data.

Moving averages (SMA_10, SMA_30) are effective features for trend smoothing.

Model evaluation metrics suggest LSTM offers lower error rates and better future prediction consistency.

**Conclusion**

LSTM outperforms Linear Regression in modeling stock prices due to its ability to learn time dependencies.

Stock price trends can be forecasted moderately well using deep learning techniques when enough historical data is available.

Combining machine learning models provides comparative insight into forecasting reliability.

 **Recommendations**
 
Add more technical indicators (e.g., RSI, MACD) for richer feature sets.

Explore ensemble models combining LSTM with other neural nets or regression.

Extend the prediction range and validate over multiple Nifty 50 stocks.

Incorporate macroeconomic indicators for multi-factor analysis.
