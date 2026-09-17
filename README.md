# Fraud Detection Streaming Engine

This project is a real-time, online machine learning system designed to detect fraudulent credit card transactions from a continuous stream of data.

## Overview
Unlike traditional batch-learning models, this system utilizes **Online Machine Learning** via the [River](https://riverml.xyz/) library. The model processes one transaction at a time, makes a prediction, and immediately updates its internal weights as soon as the true label (feedback) is available. 

To overcome the "Cold Start" (Warm-up) problem and the severe class imbalance typical of fraud datasets, the system allows for **Pre-training** on historical data before exposing the model to the live stream.

## Dataset
This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.
Please download the dataset and place the CSV file in the appropriate directory (e.g., `data/`) before running the system.

## System Architecture

The project consists of several interacting components:

- **`pretrain.py`**: Reads the historical CSV dataset and pre-trains two separate Logistic Regression pipelines (using SGD and Adam optimizers). The trained state (including the scaler) is serialized to `.pkl` files.
- **`server.py`**: A FastAPI application acting as the ML Engine. It exposes HTTP endpoints for predicting fraud probabilities (`/predict`), receiving feedback to update the model (`/feedback`), and serving real-time metrics (`/stats`).
- **`view.py`**: A Streamlit dashboard that visualizes real-time metrics, probability charts, and recent transaction logs.
- **`pos_producer.py`**: A simulator that reads the CSV and sends data to the server at a controlled rate, mimicking real POS terminals.
- **`boost.py`**: A high-speed simulation script using HTTP sessions to bombard the server with transactions, allowing for rapid evaluation and load testing.

## Getting Started

### 1. Prerequisites
Ensure you have Python installed and create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pre-training the Models
To give the system prior knowledge and avoid the warm-up penalty, pre-train the models on the historical dataset:
```bash
python pretrain.py
```
This generates `model_sgd.pkl` and `model_adam.pkl`.

### 3. Running the System
To run the system manually, you will need to open **three separate terminals** and ensure your virtual environment is activated in each one.

**Terminal 1: Start the FastAPI ML Server**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Start the Streamlit Dashboard**
```bash
streamlit run view.py
```

**Terminal 3: Start the Data Simulator**
To simulate real-time POS traffic with a slight delay:
```bash
python pos_producer.py --rows 500 --delay 0.01
```
*(Alternatively, you can run `python boost.py` to bombard the server with high-speed transactions for load testing).*

## Note on Features
The `Time` feature from the original Kaggle dataset is strictly monotonically increasing. Feeding this into an online `StandardScaler` causes catastrophic scaling artifacts when re-iterating over the dataset. Therefore, the `Time` feature is explicitly dropped across all scripts before passing data to the models.
