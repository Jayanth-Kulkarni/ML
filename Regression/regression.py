import numpy as np
class BaseRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weight = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros()
        self.bias = 0

        for _ in range(self.n_iter):
            y_predicted = self._approximation(X, self.weight, self.bias)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return self._predict(X, self.weight, self.bias)
    
    def _predict(self, X, w, b):
        raise NotImplementedError()

    def _approximation(self, X, w, b):
        raise NotImplementedError()
    
class LinReg(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, self.weight) + self.bias
         

class LogReg(BaseRegression):
    def _approximation(self, X, w, b):
        linear_approx = np.dot(X, w) + b
        return self._sigmoid(linear_approx)
    
    def _sigmoid(x):
        return 1/(1 + np.exp(-x))

    def _predict(self, X, w, b):
        linear_approx = np.dot(X, w) + b
        predictions = self._sigmoid(linear_approx)
        predicted_cls = [1 if i > 0.5 else 0 for i in predictions]
        return predicted_cls
    

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Utils
    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
        return corr ** 2

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Linear Regression
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinReg(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    accu = r2_score(y_test, predictions)
    print("Linear reg Accuracy:", accu)

    # Logistic reg
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogReg(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("Logistic reg classification accuracy:", accuracy(y_test, predictions))