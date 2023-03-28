import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    X_train_1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train)) #Add a column of 1s to the data
    X_train_1_transpose = np.transpose(X_train_1) #Transpose the matrix
    X_train_1_transpose_mult_X_train_1 = np.matmul(X_train_1_transpose, X_train_1) #Multiply the transposed matrix with the original matrix
    X_train_1_transpose_mult_X_train_1_inverse = np.linalg.pinv(X_train_1_transpose_mult_X_train_1) #Find the inverse of the matrix product
    X_train_1_transpose_mult_X_train_1_inverse_mult_X_train_1_transpose = np.matmul(X_train_1_transpose_mult_X_train_1_inverse, X_train_1_transpose) #Multiply the matrix product with the transposed matrix
    w = np.matmul(X_train_1_transpose_mult_X_train_1_inverse_mult_X_train_1_transpose, y_train) #Multiply the matrix product with the y_train matrix
    return w #Return the weights

def mse(X_train,y_train,w):
    X_train_1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train)) #Add a column of 1s to the data
    n = X_train_1.shape[0] #Get the number of rows in the matrix
    sum = 0 
    for i in range(n): #Loop through the rows in the matrix
        sum += (pred(X_train_1[i], w) - y_train[i])**2 #Calculate the sum of the squared differences between the predicted value and the actual value
    return sum/n #Return the mean of the sum

def pred(X_train,w):
    return np.dot(X_train,w) #Return the dot product of the X_train and w

def test_SciKit(X_train, X_test, Y_train, Y_test):
    LR = linear_model.LinearRegression() #Create a linear regression object
    LR.fit(X_train, Y_train) #Train the model using the training sets
    y_pred = LR.predict(X_test) #Make predictions using the testing set
    return mean_squared_error(Y_test, y_pred) #Return the mean squared error

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()
