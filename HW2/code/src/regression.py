# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# calculating time
from time import time

######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    
    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.dirname('__file__')
        f = os.path.join(dir, '..', 'data', filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]
    
    def plot(self, **kwargs) :
        """Plot data."""
        
        if 'color' not in kwargs :
            kwargs['color'] = 'b'
        
        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs) :
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression() :
    
    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param
    
    
    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """
        
        n,d = X.shape
        
        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        Phi = np.concatenate((np.ones(X.shape), X), axis=1)
        # part g: modify to create matrix for polynomial model
        m = self.m_
        poly_Phi = np.zeros((n, m+1))
        # which row to add to
        row_num = 0
        for xs in X:
            # current row
            row_c = []
            for i in range(m + 1):
                row_c.append(pow(xs[0], i))
            poly_Phi[row_num] += row_c
            row_num += 1
        Phi = np.array(poly_Phi)
                
        ### ========== TODO : END ========== ###
        
        return Phi
    
    
    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        start_time = time()
        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")

        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()
        
        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration

        # GD loop
        for t in xrange(tmax) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None :
                eta = 1 / float(1 + t)  # change this line
            else :
                eta = eta
            ### ========== TODO : END ========== ###
                
            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math
            
            # track error
            # hint: you cannot use self.predict(...) to make the predictions

            # calculate prediction of y
            y_pred = np.dot(self.coef_, X.T)
            # calculate the amount to update based on (y_pred - y)
            update_y = np.dot(y_pred - y, X)
            # perform the update with learning rate
            self.coef_ -= 2 * eta * update_y
            
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)                
            ### ========== TODO : END ========== ###
            
            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break
            
            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec
        
        print 'number of iterations: %d' % (t+1)

        end_time = time()
        return self, (t+1), (end_time - start_time)
    
    
    def fit(self, X, y, l2regularize = None ) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization
                
        Returns
        --------------------        
            self    -- an instance of self
        """
        start_time = time()
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        # compute X_T . X
        xtx_inv = np.linalg.pinv(np.dot(X.T, X))
        # set coefficients as closed-form solution using X_T . X
        self.coef_ = np.dot(np.dot(xtx_inv, X.T), y)

        end_time = time()
        return self, (end_time - start_time)
        ### ========== TODO : END ========== ###
    
    
    def predict(self, X) :
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part c: predict y
        # use wX + b as prediction
        y = np.dot(self.coef_, X.T)

        ### ========== TODO : END ========== ###
        
        return y
    
    
    def cost(self, X, y) :
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)
        # compute cost as (y_pred - y)
        squared_costs = np.square(self.predict(X) - y)
        cost = np.sum(squared_costs)
        ### ========== TODO : END ========== ###
        return cost
    
    
    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        n,d = X.shape
        # compute J
        J = self.cost(X, y)
        # RMS = sqrt(J / N)
        error = np.sqrt(J / n)
        
        ### ========== TODO : END ========== ###
        return error
    
    
    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'
        
        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')
    
    
    
    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print 'Visualizing data...'
    X_train_data = train_data.X
    y_train_data = train_data.y
    X_test_data = test_data.X
    y_test_data = test_data.y
    plot_data(X_train_data, y_train_data)
    plot_data(X_test_data, y_test_data)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print 'Investigating linear regression...'
    model = PolynomialRegression()
    model.coef_ = np.zeros(2)
    print(model.cost(X_train_data, y_train_data))

    # Perform experiments with different step sizes and make table
    step_sizes = [0.0001, 0.001, 0.01, 0.0407, None]
    for eta_ in step_sizes:
        self, num_iters, t = model.fit_GD(X_train_data, y_train_data, eta=eta_)
        cost = model.cost(X_train_data, y_train_data)
        coefficients = self.coef_
        print("Step-size: ", eta_, " Iterations: ", num_iters, " Cost: ", cost, " Coefficients: ", coefficients, " Time: ", t)

    print(" ----- ")
    print("Using closed-form")
    # Use closed form to fit model and print coefficients and cost
    self, t = model.fit(X_train_data, y_train_data)
    cost = model.cost(X_train_data, y_train_data)
    coefficients = self.coef_
    print("Cost: ", cost, " Coefficients: ", coefficients, " Time: ", t)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print 'Investigating polynomial regression...'

    exp = range(11)
    rms_train = []
    rms_test = []
    for e in exp:
        m_model = PolynomialRegression(m=e)
        # fit model with different exponents
        m_model.fit(X_train_data, y_train_data)
        # append RMS error for training and test sets for each model
        rms_train.append(m_model.rms_error(X_train_data, y_train_data))
        rms_test.append(m_model.rms_error(X_test_data, y_test_data))
    plt.plot(exp, rms_train, 'ro', label='Training Set', linestyle='dashed')
    plt.plot(exp, rms_test, 'bo', label='Test Set', linestyle='dashed')
    plt.xlabel('Exponent, m')
    plt.ylabel('RMS Error')
    plt.title('RMS Error for Different Exponents')
    plt.legend()
    plt.show()
    
    ### ========== TODO : END ========== ###
    print "Done!"

if __name__ == "__main__" :
    main()
