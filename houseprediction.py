import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Data Preprocessing
# Load your dataset, preprocess it, and split it into features and target variable (house prices)

# Sample dataset for demonstration
data = {
    'Area': [1000, 1500, 1200, 1800, 2000],
    'Bedrooms': [2, 3, 2, 4, 3],
    'Bathrooms': [1, 2, 1, 2, 2],
    'Price': [300000, 400000, 320000, 450000, 500000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Bathrooms']]  # Features
y = df['Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Building

# Linear Regression Model using scikit-learn
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Neural Network Model using TensorFlow
model = Sequential([
    Dense(32, activation='relu', input_shape=(3,)),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, verbose=0)

# Step 3: Evaluation
# Evaluate Linear Regression model
y_pred_lr = lin_reg.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print("Linear Regression Model - Mean Squared Error:", mse_lr)

# Evaluate Neural Network Model
y_pred_nn = model.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print("Neural Network Model - Mean Squared Error:", mse_nn)

# Step 4: Prediction
# Predict house prices using both models
new_data = np.array([[1300, 2, 1], [1700, 3, 2]])  # New data for prediction

# Predict with Linear Regression
predicted_prices_lr = lin_reg.predict(new_data)
print("Predicted Prices (Linear Regression):", predicted_prices_lr)

# Predict with Neural Network
predicted_prices_nn = model.predict(new_data)
print("Predicted Prices (Neural Network):", predicted_prices_nn)
