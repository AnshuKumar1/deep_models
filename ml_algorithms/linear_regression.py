import numpy as np

# m = #training examples , d = #features
# equation: W.T * X + b
# y ground truth, yhat prediction
# dim(X) = (d*m), dim(y) = (1*m), dim(W) = (1*d)

class LinearRegression():
    def __init__(self):
        self.learning_rate = 0.001
        self.iterations = 1000
        pass

    def y_hat(self, W, X):
        yhat = np.dot(W.T, X)
        return yhat

    def loss(self, yhat, y):
        L = 1 / self.m * np.sum(np.square(y - yhat))
        return L

    def gradient_descent(self, W, X, yhat, y):
        # dLdW shape would be (1*d)
        # dim(X) = (d*m), dim(yhat, yhat) = 1*m
        dLdW = 2 / self.m * np.dot(X, (yhat - y).T)
        W = W - self.learning_rate * dLdW
        return W

    def main(self, X, y):
        # apppend '1' in every training example.
        tmp = np.ones((1, X.shape[1]))
        X = np.append(X, tmp, axis=0)
        print(X.shape)

        self.m = X.shape[1]
        self.d = X.shape[0]

        W = np.zeros((self.d, 1))

        for it in range(self.iterations):
            yhat = self.y_hat(W, X)
            loss = self.loss(yhat, y)

            if (it % 10 == 0):
                print(f"Loss at iteration {it} is {loss}")
            W = self.gradient_descent(W, X, yhat, y)

        return W


if __name__ == "__main__":
    X = np.random.rand(1, 500)
    print(X.shape)
    y = 3 * X + 5 + np.random.randn(1, 500) * 0.1
    regression = LinearRegression()
    w = regression.main(X, y)