import numpy as np
import pandas as pd

# Implement analytical linear regression. To be used for Dataset 1
class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass

    # Gradient of the cost function
    def gradient(self, x, y, w):
        N, D = x.shape
        y_pred = np.dot(x, w)
        grad = np.dot(x.T, (y_pred - y))/N
        return grad

    def fit(self, x, y, optimizer = None):
        if x.ndim == 1:
            x = x[:, None]                         #add a dimension for the features

        if self.add_bias:
            x = np.column_stack([x,np.ones(x.shape[0])])    #add bias by adding a constant feature of value 1

        N,D = x.shape

        if optimizer is None:
            # analytical solution using the normal equations
            xtx = np.dot(x.T, x)
            xtx_inv = np.linalg.inv(xtx)
            xty = np.dot(x.T, y)
            self.w = np.dot(xtx_inv, xty)
            #self.w = np.linalg.lstsq(x, y)[0]          #return w for the least square difference
        else:
            w0 = np.zeros((D, y.shape[1]))                                # initialize the weights to 0
            self.w = optimizer.run(self.gradient, x, y, w0)      # run the optimizer to get the optimal weights
        return self

    def predict(self, x, w = None):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            x = np.column_stack([x,np.ones(x.shape[0])])

        if w is None:
            y_pred = x@self.w
        else:
            y_pred = x@w

        return y_pred

    # This is the cost function
    # Compute the mean squared error since the sum is effected by number of data points
    def meanSquareErrorLoss(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(0.5 * (y_true - y_pred)**2, axis=0)
    

# Implement logistic regression with gradient descent. To be used for Dataset 2
logistic = lambda z:1./ (1 + np.exp(-z))

class LogisticRegression:

    def __init__(self, add_bias=True, learning_rate=.1, epsilon=1e-4, max_iters=1e5, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients
        self.max_iters = max_iters                    #maximum number of iteration of gradient descent
        self.verbose = verbose
        self.w = None # Initialize weights to None

    def gradient(self, x, y, w):
        N,D = x.shape
        yh = logistic(np.dot(x, w))    # predictions  size N
        grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
        return grad                         # size D

    # Optimizer is Gradient Descent or Mini-batch SGD
    def fit(self, x, y, optimizer):
        if x.ndim == 1:
            x = x[:, None]

        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])


        N,D = x.shape
        self.w = np.zeros(D)

        self.w = optimizer.run(self.gradient, x, y, self.w)

        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
        return self

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]

        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))            #predict output
        return yh

    # This is the cost function
    def cost_fn(self, x, y):
        if x.ndim == 1:
            x = x[:, None]

        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        #N, D = x.shape
        z = np.dot(x, self.w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z))) #log1p calculates log(1+x) to remove floating point inaccuracies
        return J
    

class GradientDescent:

    def __init__(self, learning_rate=.1, max_iters=1e4, epsilon=1e-8, record_history=False):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.record_history = record_history
        self.epsilon = epsilon
        if record_history:
            self.w_history = []                 #to store the weight history for visualization

    def run(self, gradient_fn, x, y, w):
        grad = np.inf
        t = 1
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient_fn(x, y, w)               # compute the gradient with present weight

            if self.record_history:
                self.w_history.append(w)

            w = w - self.learning_rate * grad         # weight update step

            t += 1
        return w
    

class Mini_Batch_SGD:

    def __init__(self, learning_rate=.001, max_iters=1000, epsilon=1e-8, batch_size=None, record_history=False ):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.record_history = record_history
        if record_history:
            self.w_history = []

    def run(self, gradient_fn, x, y, w):
        N, D = x.shape
        grad = np.inf
        t = 1

        if self.batch_size is None:
            batch_size = N
        else:
            batch_size = self.batch_size

        # Convert DataFrame to numpy array if needed
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            # Random permutation of indices
            indices = np.random.permutation(N)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for i in range(0, N, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Compute gradient on current mini-batch
                grad = gradient_fn(x_batch, y_batch, w)
                # print("grad:", grad)

                if self.record_history:
                    self.w_history.append(w)

                # Update weights - now handles multiple outputs correctly
                w = w - self.learning_rate * grad

                t += 1
                if np.linalg.norm(grad) <= self.epsilon or t >= self.max_iters:
                    break

        return w