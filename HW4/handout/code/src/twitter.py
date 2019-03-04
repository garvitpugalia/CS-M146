"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        counter = 0
        # go through each line in file
        for line in fid:
            words_in_line = extract_words(line)
            # for each word in the line
            for word_t in words_in_line:
                if not word_t in word_list.keys():
                    # if a new word is found, add to the dictionary with counter
                    word_list[word_t] = counter
                    # counter keeps track of unique words
                    counter += 1
        ### ========== TODO : END ========== ###
    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        for line_num, line in enumerate(fid):
            words_in_line = extract_words(line)
            # for each word in line
            for word_t in words_in_line:
                if not word_t in word_list.keys():
                    # return error if word was not added to the dictionary
                    print("Error: The word was not found in the dictionary")
                    exit
                else:
                    word_num = word_list[word_t]
                    # set M[line][word index in dictionary] = 1
                    feature_matrix[line_num][word_num] = 1
        ### ========== TODO : END ========== ###
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc'       
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
        # uses y_pred instead of y_label as the magnitude matters
        return metrics.roc_auc_score(y_true, y_pred)
    return 0
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance
    metric_scores = []
    for train_index, test_index in kf:
        # declare a temporary classifier
        clf_t = clf
        # gather and split the datasets
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        # fit the classifier to the training data
        clf_t.fit(X_train, y_train)
        # use the classifier to predict y values
        y_values = clf_t.decision_function(X_test)
        # calculate score using performance function and append to all scores
        metric_scores.append(performance(y_test, y_values, metric=metric))
    return np.mean(metric_scores)
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print("Linear SVM Hyperparameter Selection based on " + str(metric) + ":")
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2: select optimal hyperparameter using cross-validation
    max_score = 0.0
    best_C = 0.0
    # for every value of C
    for c in C_range:
        svc_t = SVC(C=c, kernel="linear")
        metric_score = cv_performance(svc_t, X, y, kf, metric=metric)
        print("C = ", str(c), ", Metric = ", str(metric_score))
        # if a better metric score is found, replace and store the new best_C
        if metric_score > max_score:
            max_score = metric_score
            best_C = c
    return best_C
    ### ========== TODO : END ========== ###



def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 3: return performance on test data by first computing predictions and then calling performance
    y_pred = clf.decision_function(X)        
    return performance(y, y_pred, metric=metric)
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc"]
    
    ### ========== TODO : START ========== ###
    # part 1: split data into training (training + cross-validation) and testing set
    X_train_data = X[:560] # first 560
    X_test_data = X[560:] # last 70
    y_train_data = y[:560] 
    y_test_data = y[560:]
    
    # part 2: create stratified folds (5-fold CV)
    kf = StratifiedKFold(y_train_data, n_folds=5) # no shuffling used as updated by TA

    # part 2: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    for metric_t in metric_list:
        # use select_param_linear with the training data, which returns the optimal C
        C_opt = select_param_linear(X_train_data, y_train_data, kf, metric=metric_t)
        print("For metric = " + str(metric_t) + ", the optimal value of C = " + str(C_opt))

    # using the best parameters, C = 10 for acc and F1, C = 1.0 for auroc
    
    # part 3: train linear-kernel SVMs with selected hyperparameters
    clf_acc_f1 = SVC(C=10, kernel="linear")
    clf_acc_f1.fit(X_train_data, y_train_data)

    clf_auroc = SVC(C=1.0, kernel="linear")
    clf_auroc.fit(X_train_data, y_train_data)
    
    # part 3: report performance on test data
    print("For accuracy: C = 10, accuracy = " + str(performance_test(clf_acc_f1, X_test_data, y_test_data, metric="accuracy")))
    print("For F1-score: C = 10, accuracy = " + str(performance_test(clf_acc_f1, X_test_data, y_test_data, metric="f1_score")))
    print("For auroc: C = 1.0, accuracy = " + str(performance_test(clf_auroc, X_test_data, y_test_data, metric="auroc")))
    
    
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
