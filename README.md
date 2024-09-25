# Zomato Delivery Time Prediction

## Overview
This project predicts the delivery time for Zomato orders using a machine learning model. By analyzing historical data on deliveries, the model helps estimate how long a new order will take to reach the customer based on features like restaurant location, order preparation time, traffic conditions, and more. The model is aimed at optimizing customer satisfaction by providing accurate delivery time estimates.

## Performance
- **R² Score**: 0.816
- **Root Mean Squared Error (RMSE)**: 3.8 minutes

The model provides a reliable estimate of delivery time, making it a useful tool for Zomato or similar delivery-based services.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/VivekShinde7/zomato-delivery-time-prediction.git
    cd zomato-delivery-time-prediction
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the project:
    ```bash
    python app.py
    ```
####