import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

EPOCHS = 5000 #Number of epochs

def fit_perceptron(X_train, y_train):
    X_train_1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train)) #Add a column of 1s to the data
    n = X_train.shape[0] #Number of training examples
    w = np.zeros(X_train.shape[1]) #Initialize the weights to 0
    w = np.hstack((1, w)) #Add a column to w, that makes w0= 1 which is the bias
    w_opt = w #Initialize the optimal weights to the initial weights
    for e in range(EPOCHS):
        for i in range(n):
            if pred(X_train_1[i], w) != y_train[i]: #If the prediction is wrong
                w += y_train[i] * X_train_1[i] #Update the weights
                if errorPer(X_train_1, y_train, w) < errorPer(X_train_1, y_train, w_opt): #If the error is less than the optimal error
                    w_opt = w #Update the optimal weights
    return w_opt #Return the optimal weights

def errorPer(X_train,y_train,w):
    size = X_train.shape[0] #Number of training examples
    sum = 0
    for i in range(size): #Calculate the error for each training example
        sum += pred(X_train[i], w) != y_train[i] #If the prediction is wrong, add 1 to the sum
    return sum/size #Return the error percentage

def confMatrix(X_train,y_train,w):
    X_train_1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train)) #Add a column of 1s to the data
    m = np.zeros((2,2)) #Initialize the confusion matrix
    n = X_train_1.shape[0] #Number of training examples
    for i in range(n): #For each example
        if y_train[i] == -1:
            if pred(X_train_1[i], w) == -1:
                m[0][0] += 1 #If the prediction is a true negative, add 1 to the sum at index (0,0)
            else:
                m[0][1] += 1 #If the prediction is a false positive, add 1 to the sum at index (0,1)
        else:
            if pred(X_train_1[i], w) == -1:
                m[1][0] += 1 #If the prediction is a false negative, add 1 to the sum at index (1,0)
            else:
                m[1][1] += 1 #If the prediction is a true positive, add 1 to the sum at index (1,1)
    return m
    
def pred(X_train,w):
    if np.dot(X_train,w) > 0: #If the dot product is greater than 0, return 1, else return -1
        return 1
    else:
        return -1
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    clf = Perceptron(tol=1e-3, random_state=0) #Initialize the perceptron
    clf.fit(X_train, Y_train) #Fit the perceptron
    y_pred = clf.predict(X_test) #Predict the labels
    return confusion_matrix(Y_test, y_pred) #Return the confusion matrix

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True) 
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2) 

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()