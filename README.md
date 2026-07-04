# Linear Regression from Scratch

A NumPy-only implementation of Linear Regression built from scratch without using Scikit-learn's Linear Regression implementation.

The goal of this project was to understand how Linear Regression actually learns rather than simply calling a library implementation. The model follows a Scikit-learn-inspired API with `fit()`, `predict()`, and `score()` methods while implementing gradient descent manually.

## Features

- Pure NumPy implementation
- Scikit-learn-inspired API (`fit`, `predict`, `score`)
- Multiple Linear Regression support
- Batch Gradient Descent optimization
- Mean Squared Error (MSE) cost function
- Cost history tracking
- Evaluation metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

## Project Structure

```
Linear_Regression_Raw/
│
├── requirements.txt
├── model.py          # Linear Regression implementation
├── main.ipynb        # Training, testing and visualization
└── README.md

```

## How it Works

### `fit()`

The training method.

It initializes the model parameters, repeatedly performs gradient descent, and stores the learned weights and bias inside the model.

### `predict()`

Uses the learned parameters to estimate target values.

The prediction follows the linear equation:

ŷ = Xw + b

### Gradient Descent

Gradient Descent is the optimization algorithm responsible for learning.

Starting from zero weights, it repeatedly:

1. Makes predictions
2. Computes the prediction error
3. Calculates the gradients
4. Updates the weights and bias

This process is repeated for a specified number of iterations.

### Cost Function

The model minimizes Mean Squared Error (MSE) using the following cost function:

J(w, b) = (1 / 2m) * Σ(ŷ - y)²

The cost is recorded after every iteration to monitor the learning progress and visualize how the loss changes during training.

## Example

```python
model = LinearRegression(
    learning_rate=1.1,
    n_iterations=1000
)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))
```

## Dependencies

- Python
- NumPy
- Matplotlib (visualization)
- Scikit-learn (datasets and train/test split)

> **Note:** This project was built for educational purposes to understand the mathematics and optimization behind Linear Regression before relying on high-level machine learning libraries.
