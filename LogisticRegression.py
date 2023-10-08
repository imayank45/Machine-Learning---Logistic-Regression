import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class LogisitcRegression():
    
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):  
        n_samples, n_features = X.shape
        
        # initializing weights and bias as zeros
        self.weights = np.zeros(n_features)  
        self.bias = 0
        
        # prediction using using lienar regression and passing these predictions to 
        # sigmoid function 
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
        
            # calculating error
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # optimizing weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias    = self.bias    - self.lr * db 
            
    # choose the label based on probability
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred