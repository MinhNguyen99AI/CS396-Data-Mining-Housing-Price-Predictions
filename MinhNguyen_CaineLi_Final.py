import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import scipy.spatial
import timeit
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.scorer import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# ===============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")
    
    
    demonstrateHelpers(trainDF)

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperiment(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    

# ===============================================================================
#function for reading data files    
def readData():
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")
    return trainDF,testDF
#================================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)

    alg2 = KNeighborsClassifier(n_neighbors = 1)
    cvMeanScore2 = model_selection.cross_val_score(alg2, trainInput.loc[:, predictors], trainOutput, cv=2, scoring='r2', n_jobs=-1)
    print("CV Average Score:", cvMeanScore2.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(trainInput.loc[:, predictors], trainOutput, random_state = 50, test_size = 0.25)
    clf = DecisionTreeClassifier(criterion = 'entropy')
    clf.fit(X_train, y_train)
    y_pred =  clf.predict(X_test)
    print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
    print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
# ===============================================================================

'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()
    alg2 = KNeighborsClassifier(n_neighbors = 1)
    X_train, X_test, y_train, y_test = train_test_split(trainInput.loc[:, predictors], trainOutput, random_state = 50, test_size = 0.25)
    clf = DecisionTreeClassifier(criterion = 'entropy')
    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    alg2.fit(trainInput.loc[:, predictors], trainOutput)
    clf.fit(X_train, y_train)
    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])
    predictionsKNN = alg2.predict(testInput.loc[:, predictors])
    y_pred =  clf.predict(testInput.loc[:, predictors])
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })
    submissionKNN = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictionsKNN
    })
    submissionCLF = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": y_pred
    })
    # Prepare CSV
    submission.to_csv('testResults.csv', index=False)
    submissionKNN.to_csv('testResultsKNN.csv', index=False)
    submissionCLF.to_csv('testResultsCLF.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ===============================================================================
# Data cleaning - conversion, normalization

#standardize
def standardize(thatDf, listOfCols):
   # df = df.apply(lambda x: x.std(), axis = 1)
    thatDf.loc[:, listOfCols] = (thatDf.loc[:, listOfCols] - thatDf.loc[:, listOfCols].apply(lambda x: x.mean(), axis = 0))/ thatDf.loc[:, listOfCols].apply(lambda x: x.std(), axis = 0)
#normalize
def normalize(dataFrame, listOfCols):
    dataFrame.loc[:, listOfCols] = (dataFrame.loc[:, listOfCols] - dataFrame.loc[:, listOfCols].min())/(dataFrame.loc[:, listOfCols].max() - dataFrame.loc[:, listOfCols].min())
#================================================================================
'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):   
    #predictors = ['1stFlrSF','2ndFlrSF','LotShape','Street','Alley', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2','HouseStyle', 'Exterior1st', 'Exterior2nd', 'ExterQual','ExterCond', 'Foundation','BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','HeatingQC','CentralAir','Electrical','KitchenQual', 'Functional', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageCond','PavedDrive','SaleType']
    predictors = ['LotFrontage', 'FullBath', '1stFlrSF','LandSlope','2ndFlrSF','MSZoning','Street','Alley', 'LandContour', 'Neighborhood', 'Condition1','Condition2','HouseStyle','ExterQual','Foundation', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1' ,'HeatingQC', 'CentralAir','KitchenQual','Functional','FireplaceQu', 'GarageFinish', 'GarageType','GarageQual', 'GarageCond','PavedDrive','SaleType', 'GarageYrBlt','MasVnrArea', 'LotConfig','BldgType','MSSubClass','LotArea','OverallQual', 'OverallCond','YearBuilt','YearRemodAdd','KitchenAbvGr','Fireplaces','GarageYrBlt','GarageArea','GarageCars','WoodDeckSF'] 
    inputCols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
  
    #Numerics
    #LotFrontage
    trainDF.loc[:, "LotFrontage"] = trainDF.loc[:, "LotFrontage"].fillna(trainDF.loc[:, "LotFrontage"].mean())
    testDF.loc[:, "LotFrontage"] = trainDF.loc[:, "LotFrontage"].fillna(trainDF.loc[:, "LotFrontage"].mean())
    
    #GarageYrBlt
    trainDF.loc[:, "GarageYrBlt"] = trainDF.loc[:, "GarageYrBlt"].fillna(trainDF.loc[:, "GarageYrBlt"].median())
    testDF.loc[:, "GarageYrBlt"] = trainDF.loc[:, "GarageYrBlt"].fillna(trainDF.loc[:, "GarageYrBlt"].median())

    #MasVnrArea
    trainDF.loc[:, "MasVnrArea"] = trainDF.loc[:, "MasVnrArea"].fillna(trainDF.loc[:, "MasVnrArea"].mean())
    testDF.loc[:, "MasVnrArea"] = trainDF.loc[:, "MasVnrArea"].fillna(trainDF.loc[:, "MasVnrArea"].mean())
    
    
    #GarageArea
    testDF.loc[:, "GarageArea"] = trainDF.loc[:, "GarageArea"].fillna(0)

    #GarageCars
    testDF.loc[:, "GarageCars"] = trainDF.loc[:, "GarageCars"].fillna(0)   
    
    standardize(trainDF, inputCols)
    standardize(testDF, inputCols)
    
    #Non-numerics
    #Ordinal
    #Condition1
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 0 if v=='Norm' else v)
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 1 if v=='Feedr' else v)
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 2 if v=='PosN' else v)
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 3 if v=='Artery' else v)
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 4 if v=='RRAe' else v)
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 5 if v=='RRNn' else v)
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 6 if v=='RRAn' else v)
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 7 if v=='PosA' else v)
    trainDF.loc[:, "Condition1"] = trainDF.loc[:, "Condition1"].map(lambda v: 8 if v=='RRNe' else v)
    
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 0 if v=='Norm' else v)
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 1 if v=='Feedr' else v)
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 2 if v=='PosN' else v)
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 3 if v=='Artery' else v)
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 4 if v=='RRAe' else v)
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 5 if v=='RRNn' else v)
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 6 if v=='RRAn' else v)
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 7 if v=='PosA' else v)
    testDF.loc[:, "Condition1"] = testDF.loc[:, "Condition1"].map(lambda v: 8 if v=='RRNe' else v) 
    
    #Condition2
    trainDF.loc[:, "Condition2"] = trainDF.loc[:, "Condition2"].map(lambda v: 0 if v=='Norm' else v)
    trainDF.loc[:, "Condition2"] = trainDF.loc[:, "Condition2"].map(lambda v: 1 if v=='Artery' else v)
    trainDF.loc[:, "Condition2"] = trainDF.loc[:, "Condition2"].map(lambda v: 2 if v=='RRNn' else v)
    trainDF.loc[:, "Condition2"] = trainDF.loc[:, "Condition2"].map(lambda v: 3 if v=='Feedr' else v)
    trainDF.loc[:, "Condition2"] = trainDF.loc[:, "Condition2"].map(lambda v: 4 if v=='PosN' else v)
    trainDF.loc[:, "Condition2"] = trainDF.loc[:, "Condition2"].map(lambda v: 5 if v=='PosA' else v)
    trainDF.loc[:, "Condition2"] = trainDF.loc[:, "Condition2"].map(lambda v: 6 if v=='RRAn' else v)
    trainDF.loc[:, "Condition2"] = trainDF.loc[:, "Condition2"].map(lambda v: 7 if v=='RRAe' else v)
    
    testDF.loc[:, "Condition2"] = testDF.loc[:, "Condition2"].map(lambda v: 0 if v=='Norm' else v)
    testDF.loc[:, "Condition2"] = testDF.loc[:, "Condition2"].map(lambda v: 1 if v=='Artery' else v)
    testDF.loc[:, "Condition2"] = testDF.loc[:, "Condition2"].map(lambda v: 2 if v=='RRNn' else v)
    testDF.loc[:, "Condition2"] = testDF.loc[:, "Condition2"].map(lambda v: 3 if v=='Feedr' else v)
    testDF.loc[:, "Condition2"] = testDF.loc[:, "Condition2"].map(lambda v: 4 if v=='PosN' else v)
    testDF.loc[:, "Condition2"] = testDF.loc[:, "Condition2"].map(lambda v: 5 if v=='PosA' else v)
    testDF.loc[:, "Condition2"] = testDF.loc[:, "Condition2"].map(lambda v: 6 if v=='RRAn' else v)
    testDF.loc[:, "Condition2"] = testDF.loc[:, "Condition2"].map(lambda v: 7 if v=='RRAe' else v)
   
    #BsmtQual
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 0 if v=='Gd' else v)
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 1 if v=='TA' else v)
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 2 if v=='Ex' else v)
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 3 if v=='Fa' else v)
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].fillna(trainDF.loc[:, "BsmtQual"].mode().iloc[0])
    
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].map(lambda v: 0 if v=='Gd' else v)
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].map(lambda v: 1 if v=='TA' else v)
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].map(lambda v: 2 if v=='Ex' else v)
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].map(lambda v: 3 if v=='Fa' else v)
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].fillna(trainDF.loc[:, "BsmtQual"].mode().iloc[0])
    
    #BsmtCond
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 0 if v=='Gd' else v)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 1 if v=='TA' else v)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 2 if v=='Po' else v)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 3 if v=='Fa' else v)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].fillna(trainDF.loc[:, "BsmtCond"].mode().iloc[0])
    
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].map(lambda v: 0 if v=='Gd' else v)
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].map(lambda v: 1 if v=='TA' else v)
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].map(lambda v: 2 if v=='Po' else v)
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].map(lambda v: 3 if v=='Fa' else v)
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].fillna(trainDF.loc[:, "BsmtCond"].mode().iloc[0])
   
    #BsmtExposure
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 0 if v=='No' else v)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 1 if v=='Gd' else v)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 2 if v=='Mn' else v)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 3 if v=='Av' else v)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].fillna(trainDF.loc[:, "BsmtExposure"].mode().iloc[0])
    
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].map(lambda v: 0 if v=='No' else v)
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].map(lambda v: 1 if v=='Gd' else v)
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].map(lambda v: 2 if v=='Mn' else v)
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].map(lambda v: 3 if v=='Av' else v)
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].fillna(trainDF.loc[:, "BsmtExposure"].mode().iloc[0])
    
    #HeatingQC
    trainDF.loc[:, "HeatingQC"] = trainDF.loc[:, "HeatingQC"].map(lambda v: 0 if v=='Ex' else v)
    trainDF.loc[:, "HeatingQC"] = trainDF.loc[:, "HeatingQC"].map(lambda v: 1 if v=='Gd' else v)
    trainDF.loc[:, "HeatingQC"] = trainDF.loc[:, "HeatingQC"].map(lambda v: 2 if v=='TA' else v)
    trainDF.loc[:, "HeatingQC"] = trainDF.loc[:, "HeatingQC"].map(lambda v: 3 if v=='Fa' else v)
    trainDF.loc[:, "HeatingQC"] = trainDF.loc[:, "HeatingQC"].map(lambda v: 4 if v=='Po' else v)
    
    testDF.loc[:, "HeatingQC"] = testDF.loc[:, "HeatingQC"].map(lambda v: 0 if v=='Ex' else v)
    testDF.loc[:, "HeatingQC"] = testDF.loc[:, "HeatingQC"].map(lambda v: 1 if v=='Gd' else v)
    testDF.loc[:, "HeatingQC"] = testDF.loc[:, "HeatingQC"].map(lambda v: 2 if v=='TA' else v)
    testDF.loc[:, "HeatingQC"] = testDF.loc[:, "HeatingQC"].map(lambda v: 3 if v=='Fa' else v)
    testDF.loc[:, "HeatingQC"] = testDF.loc[:, "HeatingQC"].map(lambda v: 4 if v=='Po' else v)

    #KitchenQual
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 0 if v=='Gd' else v)
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 1 if v=='TA' else v)
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 2 if v=='Ex' else v)
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 3 if v=='Fa' else v)
    
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].map(lambda v: 0 if v=='Gd' else v)
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].map(lambda v: 1 if v=='TA' else v)
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].map(lambda v: 2 if v=='Ex' else v)
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].map(lambda v: 3 if v=='Fa' else v)
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].fillna(trainDF.loc[:, "KitchenQual"].mode().iloc[0])
    
    #Functional
    trainDF.loc[:, "Functional"] = trainDF.loc[:, "Functional"].map(lambda v: 0 if v=='Typ' else v)
    trainDF.loc[:, "Functional"] = trainDF.loc[:, "Functional"].map(lambda v: 1 if v=='Min1' else v)
    trainDF.loc[:, "Functional"] = trainDF.loc[:, "Functional"].map(lambda v: 2 if v=='Maj1' else v)
    trainDF.loc[:, "Functional"] = trainDF.loc[:, "Functional"].map(lambda v: 3 if v=='Min2' else v)
    trainDF.loc[:, "Functional"] = trainDF.loc[:, "Functional"].map(lambda v: 4 if v=='Mod' else v)
    trainDF.loc[:, "Functional"] = trainDF.loc[:, "Functional"].map(lambda v: 5 if v=='Maj2' else v)
    trainDF.loc[:, "Functional"] = trainDF.loc[:, "Functional"].map(lambda v: 6 if v=='Sev' else v)
    
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].map(lambda v: 0 if v=='Typ' else v)
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].map(lambda v: 1 if v=='Min1' else v)
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].map(lambda v: 2 if v=='Maj1' else v)
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].map(lambda v: 3 if v=='Min2' else v)
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].map(lambda v: 4 if v=='Mod' else v)
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].map(lambda v: 5 if v=='Maj2' else v)
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].map(lambda v: 6 if v=='Sev' else v)
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].fillna(trainDF.loc[:, "Functional"].mode().iloc[0])
    
    #FireplaceQu
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 0 if v=='TA' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 1 if v=='Gd' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 2 if v=='Fa' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 3 if v=='Ex' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 4 if v=='Po' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].fillna(trainDF.loc[:, "FireplaceQu"].mode().iloc[0])
    
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 0 if v=='TA' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 1 if v=='Gd' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 2 if v=='Fa' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 3 if v=='Ex' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 4 if v=='Po' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].fillna(trainDF.loc[:, "FireplaceQu"].mode().iloc[0])
    
    #GarageCond
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 0 if v=='TA' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 1 if v=='Fa' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 2 if v=='Gd' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 3 if v=='Ex' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 4 if v=='Po' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].fillna(trainDF.loc[:, "GarageCond"].mode().iloc[0])
    
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 0 if v=='TA' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 1 if v=='Fa' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 2 if v=='Gd' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 3 if v=='Ex' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 4 if v=='Po' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].fillna(trainDF.loc[:, "GarageCond"].mode().iloc[0])
    
    #Nominal Data
    testDF.loc[:, "MSZoning"] = trainDF.loc[:, "MSZoning"].fillna(trainDF.loc[:, "MSZoning"].mode().iloc[0])
    testDF.loc[:, "Alley"] = trainDF.loc[:, "Alley"].fillna(trainDF.loc[:, "Alley"].mode().iloc[0])
    trainDF.loc[:, "Alley"] = trainDF.loc[:, "Alley"].fillna(trainDF.loc[:, "Alley"].mode().iloc[0])
    trainDF.loc[:, "MasVnrType"] = trainDF.loc[:, "MasVnrType"].fillna(trainDF.loc[:, "MasVnrType"].mode().iloc[0])
    testDF.loc[:, "MasVnrType"] = trainDF.loc[:, "MasVnrType"].fillna(trainDF.loc[:, "MasVnrType"].mode().iloc[0])

    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].fillna(trainDF.loc[:, "BsmtFinType1"].mode().iloc[0])
    testDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].fillna(trainDF.loc[:, "BsmtFinType1"].mode().iloc[0])
    trainDF.loc[:, "Electrical"] = trainDF.loc[:, "Electrical"].fillna(trainDF.loc[:, "Electrical"].mode().iloc[0])
    testDF.loc[:, "Electrical"] = trainDF.loc[:, "Electrical"].fillna(trainDF.loc[:, "Electrical"].mode().iloc[0])
   
    trainDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].fillna(trainDF.loc[:, "GarageType"].mode().iloc[0])
    testDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].fillna(trainDF.loc[:, "GarageType"].mode().iloc[0])
    trainDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].fillna(trainDF.loc[:, "GarageQual"].mode().iloc[0])
    testDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].fillna(trainDF.loc[:, "GarageQual"].mode().iloc[0])    
    trainDF.loc[:, "GarageFinish"] = trainDF.loc[:, "GarageFinish"].fillna(trainDF.loc[:, "GarageFinish"].mode().iloc[0])
    testDF.loc[:, "GarageFinish"] = trainDF.loc[:, "GarageFinish"].fillna(trainDF.loc[:, "GarageFinish"].mode().iloc[0])

    trainDF.loc[:, "PoolQC"] = trainDF.loc[:, "PoolQC"].fillna(trainDF.loc[:, "PoolQC"].mode().iloc[0])
    testDF.loc[:, "PoolQC"] = trainDF.loc[:, "PoolQC"].fillna(trainDF.loc[:, "PoolQC"].mode().iloc[0])
    trainDF.loc[:, "Fence"] = trainDF.loc[:, "Fence"].fillna(trainDF.loc[:, "PoolQC"].mode().iloc[0])
    testDF.loc[:, "Fence"] = trainDF.loc[:, "Fence"].fillna(trainDF.loc[:, "Fence"].mode().iloc[0])
    trainDF.loc[:, "MiscFeature"] = trainDF.loc[:, "MiscFeature"].fillna(trainDF.loc[:, "MiscFeature"].mode().iloc[0])
    testDF.loc[:, "MiscFeature"] = trainDF.loc[:, "MiscFeature"].fillna(trainDF.loc[:, "MiscFeature"].mode().iloc[0])
    testDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].fillna(trainDF.loc[:, "SaleType"].mode().iloc[0])
    
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].fillna(trainDF.loc[:, "HouseStyle"].mode().iloc[0])
    testDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].fillna(trainDF.loc[:, "HouseStyle"].mode().iloc[0])
    
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].fillna(trainDF.loc[:, "BsmtExposure"].mode().iloc[0])
    testDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].fillna(trainDF.loc[:, "BsmtExposure"].mode().iloc[0])
    #get_Dummies
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]

    trainInput = pd.get_dummies(trainInput, columns=['MSZoning','LandSlope','Street','Alley', 'LandContour', 'Neighborhood','BsmtFinType1', 'CentralAir','GarageType','PavedDrive','SaleType', 'LotConfig','BldgType', "Foundation", "ExterQual", "GarageQual", "GarageFinish", "HouseStyle", "BsmtExposure"])
    testInput = pd.get_dummies(testInput, columns=['MSZoning','LandSlope','Street','Alley', 'LandContour', 'Neighborhood','BsmtFinType1', 'CentralAir','GarageType','PavedDrive','SaleType','LotConfig','BldgType', "Foundation", "ExterQual", "GarageQual", "GarageFinish", "HouseStyle", "BsmtExposure"])
    
    predictors = trainInput.columns

    print("Predictors:", predictors)

    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

