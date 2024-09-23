# time_series_forecasting

Overview
This repository contains code and resources for time series forecasting using various machine learning techniques. The goal is to predict future values based on historical data, leveraging models such as LSTM neural networks, ARIMA, and ensemble methods.

Features
Data preprocessing and feature engineering
Implementation of multiple forecasting models
LSTM neural networks
ARIMA
XGBoost and other ensemble methods
Model evaluation and comparison
Visualization of results
Datasets
One of the main datasets used in this project is customer billing data. To ensure privacy and reproducibility, the repository includes a script to create a pseudo dataset that mimics the characteristics of the original customer billing data.

Installation
To get started, clone the repository and install the required dependencies:

git clone https://github.com/RonKlem/time_series_forecasting.git
cd time_series_forecasting
pip install -r requirements.txt

Usage

Data Wrangling: Run the data wrangling script to clean and prepare the data.
python customer_billing_data_wrangling.ipynb

Model Training: Train the forecasting models using the prepared data.
python Forecasting_POC.ipynb

(not available yet)
Evaluation: Evaluate the performance of the models and visualize the results.
python evaluate.py

(not available yet)
Results
The results of the forecasting models, including performance metrics and visualizations, will be saved in the results/ directory.

Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

Contact
For any questions or inquiries, please contact Ron Klem at rklem88@yahoo.com
