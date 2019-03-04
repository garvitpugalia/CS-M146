"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        
        c = Counter(y)
        num_positive = c[1]
        num_negative = c[0]
        total_samples = float(num_positive + num_negative)

        self.probabilities_ = {0: num_negative/total_samples, 1: num_positive/total_samples}
        
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        
        num_predictions = len(X)
        probability_dist = [self.probabilities_[0], self.probabilities_[1]]

        y = np.random.choice(2, num_predictions, p=probability_dist)
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0

    for trial_num in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=trial_num)
        clf.fit(X_train, y_train)
        y_train_clf = clf.predict(X_train)
        y_test_clf = clf.predict(X_test)

        train_error = train_error + (1 - metrics.accuracy_score(y_train, y_train_clf))
        test_error = test_error + (1 - metrics.accuracy_score(y_test, y_test_clf))

    train_error = train_error/float(ntrials)
    test_error = test_error/float(ntrials)
        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier

    print('Classifying using Random...')
    clf = RandomClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred)
    print('\t-- training error: %.3f' % train_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 

    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred)
    print('\t-- training error: %.3f' % train_error)
    
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 

    print('Classifying using k-Nearest Neighbors...')
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred)
    print('\t-- training error: %.3f' % train_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers

    print('Investigating various classifiers...')
    train_error, test_error = error(MajorityVoteClassifier(), X, y, 100, 0.2)
    print('Cross-validation error for MajorityVoteClassifier')
    print('\t-- training error: %.3f' % train_error)
    print('\t-- test error: %.3f' % test_error)
    train_error, test_error = error(RandomClassifier(), X, y, 100, 0.2)
    print('Cross-validation error for RandomClassifier')
    print('\t-- training error: %.3f' % train_error)
    print('\t-- test error: %.3f' % test_error)
    train_error, test_error = error(DecisionTreeClassifier(criterion='entropy'), X, y, 100, 0.2)
    print('Cross-validation error for DecisionTreeClassifier')
    print('\t-- training error: %.3f' % train_error)
    print('\t-- test error: %.3f' % test_error)
    train_error, test_error = error(KNeighborsClassifier(n_neighbors=5), X, y, 100, 0.2)
    print('Cross-validation error for kNeighborsClassifier with k = 5')
    print('\t-- training error: %.3f' % train_error)
    print('\t-- test error: %.3f' % test_error)
    
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier

    print('Finding the best k for KNeighbors classifier...')
    avg_validation_error = []
    for k in range(1, 50, 2):
        error_list = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=10)
        avg_validation_error.append(np.mean(error_list))
    plt.xlabel('k')
    plt.ylabel('Cross-validation score')
    plt.title('Error per number of neighbors in KNeighborsClassifier')
    plt.plot(list(range(1, 50, 2)), avg_validation_error, '+', linestyle='-')
    plt.show()
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths

    print('Investigating depths...')
    train_error_list = []
    test_error_list = []
    for depth in range(1, 20):
        train_error, test_error = error(DecisionTreeClassifier(criterion='entropy', max_depth=depth), X, y, 100, 0.2)
        train_error_list.append(train_error)
        test_error_list.append(test_error)
    plt.xlabel('Maximum depth')
    plt.ylabel('Average error')
    plt.title('Average training and test error per maximum depth in DecisionTreeClassifier')
    plt.plot(list(range(1,20)), train_error_list, '+', linestyle=':', label='train error')
    plt.plot(list(range(1,20)), test_error_list, 'o', linestyle='-', label='test error')
    plt.legend()
    plt.show()
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes

    print('Investigating training set sizes...')
    decision_tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    neighbor_clf = KNeighborsClassifier(n_neighbors=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    decision_train_errors = []
    decision_test_errors = []
    neighbor_train_errors = []
    neighbor_test_errors = []

    training_split = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for split in training_split:
        X_train_frac, X_test_frac, y_train_frac, y_test_frac = train_test_split(X_train, y_train, train_size = split)

        decision_tree_clf.fit(X_train_frac, y_train_frac)
        pred_y_train = decision_tree_clf.predict(X_train)
        pred_y_test = decision_tree_clf.predict(X_test_frac)
        decision_train_errors.append(1 - metrics.accuracy_score(y_train, pred_y_train))
        decision_test_errors.append(1 - metrics.accuracy_score(y_test_frac, pred_y_test))

        neighbor_clf.fit(X_train_frac, y_train_frac)
        pred_y_train = neighbor_clf.predict(X_train)
        pred_y_test = neighbor_clf.predict(X_test_frac)
        neighbor_train_errors.append(1 - metrics.accuracy_score(y_train, pred_y_train))
        neighbor_test_errors.append(1 - metrics.accuracy_score(y_test_frac, pred_y_test))

    decision_tree_clf.fit(X_train, y_train)
    pred_y_train = decision_tree_clf.predict(X_train)
    pred_y_test = decision_tree_clf.predict(X_test_frac)
    decision_train_errors.append(1 - metrics.accuracy_score(y_train, pred_y_train))
    decision_test_errors.append(1 - metrics.accuracy_score(y_test_frac, pred_y_test))

    neighbor_clf.fit(X_train, y_train)
    pred_y_train = neighbor_clf.predict(X_train)
    pred_y_test = neighbor_clf.predict(X_test_frac)
    neighbor_train_errors.append(1 - metrics.accuracy_score(y_train, pred_y_train))
    neighbor_test_errors.append(1 - metrics.accuracy_score(y_test_frac, pred_y_test))

    training_split.append(1)

    plt.ylim(bottom=0,top=1)
    plt.xlabel('Training split fraction')
    plt.ylabel('Error')
    plt.title('Training and test error for different training set splits in classifier')
    plt.plot(training_split, decision_train_errors, 'r+', linestyle=':', label='decision-tree training set')
    plt.plot(training_split, decision_test_errors, 'ro', linestyle='-', label='decision-tree test set')
    plt.plot(training_split, neighbor_train_errors, 'b+', linestyle=':', label='neighbor training set')
    plt.plot(training_split, neighbor_test_errors, 'bo', linestyle='-', label='neighbor test set')
    plt.legend()
    plt.show()
    
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
