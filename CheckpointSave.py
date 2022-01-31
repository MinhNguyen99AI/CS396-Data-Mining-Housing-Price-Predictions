import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")

    demonstrateHelpers(trainDF)

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperiment(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)

# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)

    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    predictors = ['1stFlrSF','2ndFlrSF','LotShape','Street','Alley', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2','HouseStyle', 'Exterior1st', 'Exterior2nd', 'ExterQual','ExterCond', 'Foundation','BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','HeatingQC','CentralAir','Electrical','KitchenQual', 'Functional', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageCond','PavedDrive','SaleType'  ]
    #predictors = ['LotShape']
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    #Numeric Data
    #LotFrontage
    trainDF.loc[:, "LotFrontage"] = trainDF.loc[:, "LotFrontage"].fillna(trainDF.loc[:, "LotFrontage"].mean())
    testDF.loc[:, "LotFrontage"] = trainDF.loc[:, "LotFrontage"].fillna(trainDF.loc[:, "LotFrontage"].mean())
    
    #GarageYrBlt
    trainDF.loc[:, "GarageYrBlt"] = trainDF.loc[:, "GarageYrBlt"].fillna(trainDF.loc[:, "GarageYrBlt"].median())
    testDF.loc[:, "GarageYrBlt"] = trainDF.loc[:, "GarageYrBlt"].fillna(trainDF.loc[:, "GarageYrBlt"].median())
    
    #MasVnrArea
    trainDF.loc[:, "MasVnrArea"] = trainDF.loc[:, "MasVnrArea"].fillna(trainDF.loc[:, "MasVnrArea"].mean())
    testDF.loc[:, "MasVnrArea"] = trainDF.loc[:, "MasVnrArea"].fillna(trainDF.loc[:, "MasVnrArea"].mean())
    
    
    #Non-Numeric Data
    testDF.loc[:, "MSZoning"] = trainDF.loc[:, "MSZoning"].fillna(trainDF.loc[:, "MSZoning"].mode().iloc[0])
    testDF.loc[:, "Alley"] = trainDF.loc[:, "Alley"].fillna(trainDF.loc[:, "Alley"].mode().iloc[0])
    trainDF.loc[:, "Alley"] = trainDF.loc[:, "Alley"].fillna(trainDF.loc[:, "Alley"].mode().iloc[0])
    testDF.loc[:, "Utilities"] = trainDF.loc[:, "Utilities"].fillna(trainDF.loc[:, "Utilities"].mode().iloc[0])
    testDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].fillna(trainDF.loc[:, "Exterior1st"].mode().iloc[0])
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].fillna(trainDF.loc[:, "Exterior2nd"].mode().iloc[0])
    trainDF.loc[:, "MasVnrType"] = trainDF.loc[:, "MasVnrType"].fillna(4)

    #LotShape
    
    trainDF.loc[:, "LotShape"] = trainDF.loc[:, "LotShape"].map(lambda v: 0 if v=='Reg' else v)
    trainDF.loc[:, "LotShape"] = trainDF.loc[:, "LotShape"].map(lambda v: 1 if v=='IR1' else v)
    trainDF.loc[:, "LotShape"] = trainDF.loc[:, "LotShape"].map(lambda v: 2 if v=='IR2' else v)
    trainDF.loc[:, "LotShape"] = trainDF.loc[:, "LotShape"].map(lambda v: 3 if v=='IR3' else v)
    
    testDF.loc[:, "LotShape"] = testDF.loc[:, "LotShape"].map(lambda v: 0 if v=='Reg' else v)
    testDF.loc[:, "LotShape"] = testDF.loc[:, "LotShape"].map(lambda v: 1 if v=='IR1' else v)
    testDF.loc[:, "LotShape"] = testDF.loc[:, "LotShape"].map(lambda v: 2 if v=='IR2' else v)
    testDF.loc[:, "LotShape"] = testDF.loc[:, "LotShape"].map(lambda v: 3 if v=='IR3' else v)
    
    #MSZoning
    trainDF.loc[:, "MSZoning"] = trainDF.loc[:, "MSZoning"].map(lambda v: 0 if v=='RL' else v)
    trainDF.loc[:, "MSZoning"] = trainDF.loc[:, "MSZoning"].map(lambda v: 1 if v=='RM' else v)
    trainDF.loc[:, "MSZoning"] = trainDF.loc[:, "MSZoning"].map(lambda v: 2 if v=='C (all)' else v)
    trainDF.loc[:, "MSZoning"] = trainDF.loc[:, "MSZoning"].map(lambda v: 3 if v=='FV' else v)
    trainDF.loc[:, "MSZoning"] = trainDF.loc[:, "MSZoning"].map(lambda v: 4 if v=='RH' else v)
    
    testDF.loc[:, "MSZoning"] = testDF.loc[:, "MSZoning"].map(lambda v: 0 if v=='RL' else v)
    testDF.loc[:, "MSZoning"] = testDF.loc[:, "MSZoning"].map(lambda v: 1 if v=='RM' else v)
    testDF.loc[:, "MSZoning"] = testDF.loc[:, "MSZoning"].map(lambda v: 2 if v=='C (all)' else v)
    testDF.loc[:, "MSZoning"] = testDF.loc[:, "MSZoning"].map(lambda v: 3 if v=='FV' else v)
    testDF.loc[:, "MSZoning"] = testDF.loc[:, "MSZoning"].map(lambda v: 4 if v=='RH' else v)
    testDF.loc[:, "MSZoning"] = trainDF.loc[:, "MSZoning"].fillna(trainDF.loc[:, "MSZoning"].mode().iloc[0])
    
    #Street
    trainDF.loc[:, "Street"] = trainDF.loc[:, "Street"].map(lambda v: 0 if v=='Pave' else v)
    trainDF.loc[:, "Street"] = trainDF.loc[:, "Street"].map(lambda v: 1 if v=='Grvl' else v)
    
    testDF.loc[:, "Street"] = testDF.loc[:, "Street"].map(lambda v: 0 if v=='Pave' else v)
    testDF.loc[:, "Street"] = testDF.loc[:, "Street"].map(lambda v: 1 if v=='Grvl' else v)
    
    #Alley
    trainDF.loc[:, "Alley"] = trainDF.loc[:, "Alley"].map(lambda v: 0 if v=='Pave' else v)
    trainDF.loc[:, "Alley"] = trainDF.loc[:, "Alley"].map(lambda v: 1 if v=='Grvl' else v)
    trainDF.loc[:, "Alley"] = trainDF.loc[:, "Alley"].fillna(2)
    
    testDF.loc[:, "Alley"] = testDF.loc[:, "Alley"].map(lambda v: 0 if v=='Pave' else v)
    testDF.loc[:, "Alley"] = testDF.loc[:, "Alley"].map(lambda v: 1 if v=='Grvl' else v)
    testDF.loc[:, "Alley"] = testDF.loc[:, "Alley"].fillna(2)
    
    #LandContour
    trainDF.loc[:, "LandContour"] = trainDF.loc[:, "LandContour"].map(lambda v: 0 if v=='Lvl' else v)
    trainDF.loc[:, "LandContour"] = trainDF.loc[:, "LandContour"].map(lambda v: 1 if v=='Bnk' else v)
    trainDF.loc[:, "LandContour"] = trainDF.loc[:, "LandContour"].map(lambda v: 2 if v=='Low' else v)
    trainDF.loc[:, "LandContour"] = trainDF.loc[:, "LandContour"].map(lambda v: 3 if v=='HLS' else v)
    
    testDF.loc[:, "LandContour"] = testDF.loc[:, "LandContour"].map(lambda v: 0 if v=='Lvl' else v)
    testDF.loc[:, "LandContour"] = testDF.loc[:, "LandContour"].map(lambda v: 1 if v=='Bnk' else v)
    testDF.loc[:, "LandContour"] = testDF.loc[:, "LandContour"].map(lambda v: 2 if v=='Low' else v)
    testDF.loc[:, "LandContour"] = testDF.loc[:, "LandContour"].map(lambda v: 3 if v=='HLS' else v)
    
    #Ultilities
    trainDF.loc[:, "Utilities"] = trainDF.loc[:, "Utilities"].map(lambda v: 0 if v=='AllPub' else v)
    trainDF.loc[:, "Utilities"] = trainDF.loc[:, "Utilities"].map(lambda v: 1 if v=='NoSeWa' else v)
    
    testDF.loc[:, "Utilities"] = testDF.loc[:, "Utilities"].map(lambda v: 0 if v=='AllPub' else v)
    testDF.loc[:, "Utilities"] = testDF.loc[:, "Utilities"].map(lambda v: 1 if v=='NoSeWa' else v)
    testDF.loc[:, "Utilities"] = testDF.loc[:, "Utilities"].fillna(trainDF.loc[:, "Utilities"].mode().iloc[0])
    
    #LotConfig
    trainDF.loc[:, "LotConfig"] = trainDF.loc[:, "LotConfig"].map(lambda v: 0 if v=='Inside' else v)
    trainDF.loc[:, "LotConfig"] = trainDF.loc[:, "LotConfig"].map(lambda v: 1 if v=='FR2' else v)
    trainDF.loc[:, "LotConfig"] = trainDF.loc[:, "LotConfig"].map(lambda v: 2 if v=='Corner' else v)
    trainDF.loc[:, "LotConfig"] = trainDF.loc[:, "LotConfig"].map(lambda v: 3 if v=='CulDSac' else v)
    trainDF.loc[:, "LotConfig"] = trainDF.loc[:, "LotConfig"].map(lambda v: 4 if v=='FR3' else v)
    
    testDF.loc[:, "LotConfig"] = testDF.loc[:, "LotConfig"].map(lambda v: 0 if v=='Inside' else v)
    testDF.loc[:, "LotConfig"] = testDF.loc[:, "LotConfig"].map(lambda v: 1 if v=='FR2' else v)
    testDF.loc[:, "LotConfig"] = testDF.loc[:, "LotConfig"].map(lambda v: 2 if v=='Corner' else v)
    testDF.loc[:, "LotConfig"] = testDF.loc[:, "LotConfig"].map(lambda v: 3 if v=='CulDSac' else v)
    testDF.loc[:, "LotConfig"] = testDF.loc[:, "LotConfig"].map(lambda v: 4 if v=='FR3' else v)
    
    #LandSlope
    trainDF.loc[:, "LandSlope"] = trainDF.loc[:, "LandSlope"].map(lambda v: 0 if v=='Gtl' else v)
    trainDF.loc[:, "LandSlope"] = trainDF.loc[:, "LandSlope"].map(lambda v: 1 if v=='Mod' else v)
    trainDF.loc[:, "LandSlope"] = trainDF.loc[:, "LandSlope"].map(lambda v: 2 if v=='Sev' else v)
    
    testDF.loc[:, "LandSlope"] = testDF.loc[:, "LandSlope"].map(lambda v: 0 if v=='Gtl' else v)
    testDF.loc[:, "LandSlope"] = testDF.loc[:, "LandSlope"].map(lambda v: 1 if v=='Mod' else v)
    testDF.loc[:, "LandSlope"] = testDF.loc[:, "LandSlope"].map(lambda v: 2 if v=='Sev' else v)
    
    #Neighborhood
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 0 if v=='CollgCr' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 1 if v=='Veenker' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 2 if v=='Crawfor' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 3 if v=='NoRidge' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 4 if v=='Mitchel' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 5 if v=='Somerst' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 6 if v=='NWAmes' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 7 if v=='OldTown' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 8 if v=='BrkSide' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 9 if v=='Sawyer' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 10 if v=='NridgHt' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 11 if v=='NAmes' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 12 if v=='SawyerW' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 13 if v=='IDOTRR' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 14 if v=='MeadowV' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 15 if v=='Edwards' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 16 if v=='Timber' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 17 if v=='Gilbert' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 18 if v=='StoneBr' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 19 if v=='ClearCr' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 20 if v=='NPkVill' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 21 if v=='Blmngtn' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 22 if v=='BrDale' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 23 if v=='SWISU' else v)
    trainDF.loc[:, "Neighborhood"] = trainDF.loc[:, "Neighborhood"].map(lambda v: 24 if v=='Blueste' else v)
    
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 0 if v=='CollgCr' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 1 if v=='Veenker' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 2 if v=='Crawfor' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 3 if v=='NoRidge' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 4 if v=='Mitchel' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 5 if v=='Somerst' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 6 if v=='NWAmes' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 7 if v=='OldTown' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 8 if v=='BrkSide' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 9 if v=='Sawyer' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 10 if v=='NridgHt' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 11 if v=='NAmes' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 12 if v=='SawyerW' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 13 if v=='IDOTRR' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 14 if v=='MeadowV' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 15 if v=='Edwards' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 16 if v=='Timber' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 17 if v=='Gilbert' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 18 if v=='StoneBr' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 19 if v=='ClearCr' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 20 if v=='NPkVill' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 21 if v=='Blmngtn' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 22 if v=='BrDale' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 23 if v=='SWISU' else v)
    testDF.loc[:, "Neighborhood"] = testDF.loc[:, "Neighborhood"].map(lambda v: 24 if v=='Blueste' else v)
    
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
    
    #BldgType
    trainDF.loc[:, "BldgType"] = trainDF.loc[:, "BldgType"].map(lambda v: 0 if v=='1Fam' else v)
    trainDF.loc[:, "BldgType"] = trainDF.loc[:, "BldgType"].map(lambda v: 1 if v=='2fmCon' else v)
    trainDF.loc[:, "BldgType"] = trainDF.loc[:, "BldgType"].map(lambda v: 2 if v=='Duplex' else v)
    trainDF.loc[:, "BldgType"] = trainDF.loc[:, "BldgType"].map(lambda v: 3 if v=='TwnhsE' else v)
    trainDF.loc[:, "BldgType"] = trainDF.loc[:, "BldgType"].map(lambda v: 4 if v=='Twnhs' else v)
    
    testDF.loc[:, "BldgType"] = testDF.loc[:, "BldgType"].map(lambda v: 0 if v=='1Fam' else v)
    testDF.loc[:, "BldgType"] = testDF.loc[:, "BldgType"].map(lambda v: 1 if v=='2fmCon' else v)
    testDF.loc[:, "BldgType"] = testDF.loc[:, "BldgType"].map(lambda v: 2 if v=='Duplex' else v)
    testDF.loc[:, "BldgType"] = testDF.loc[:, "BldgType"].map(lambda v: 3 if v=='TwnhsE' else v)
    testDF.loc[:, "BldgType"] = testDF.loc[:, "BldgType"].map(lambda v: 4 if v=='Twnhs' else v)
    
    #HouseStyle
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].map(lambda v: 0 if v=='2Story' else v)
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].map(lambda v: 1 if v=='1Story' else v)
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].map(lambda v: 2 if v=='1.5Fin' else v)
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].map(lambda v: 3 if v=='1.5Unf' else v)
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].map(lambda v: 4 if v=='SFoyer' else v)
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].map(lambda v: 5 if v=='SLvl' else v)
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].map(lambda v: 6 if v=='2.5Unf' else v)
    trainDF.loc[:, "HouseStyle"] = trainDF.loc[:, "HouseStyle"].map(lambda v: 7 if v=='2.5Fin' else v)
    
    testDF.loc[:, "HouseStyle"] = testDF.loc[:, "HouseStyle"].map(lambda v: 0 if v=='2Story' else v)
    testDF.loc[:, "HouseStyle"] = testDF.loc[:, "HouseStyle"].map(lambda v: 1 if v=='1Story' else v)
    testDF.loc[:, "HouseStyle"] = testDF.loc[:, "HouseStyle"].map(lambda v: 2 if v=='1.5Fin' else v)
    testDF.loc[:, "HouseStyle"] = testDF.loc[:, "HouseStyle"].map(lambda v: 3 if v=='1.5Unf' else v)
    testDF.loc[:, "HouseStyle"] = testDF.loc[:, "HouseStyle"].map(lambda v: 4 if v=='SFoyer' else v)
    testDF.loc[:, "HouseStyle"] = testDF.loc[:, "HouseStyle"].map(lambda v: 5 if v=='SLvl' else v)
    testDF.loc[:, "HouseStyle"] = testDF.loc[:, "HouseStyle"].map(lambda v: 6 if v=='2.5Unf' else v)
    testDF.loc[:, "HouseStyle"] = testDF.loc[:, "HouseStyle"].map(lambda v: 7 if v=='2.5Fin' else v)
    
    #RoofStyle
    trainDF.loc[:, "RoofStyle"] = trainDF.loc[:, "RoofStyle"].map(lambda v: 0 if v=='Gable' else v)
    trainDF.loc[:, "RoofStyle"] = trainDF.loc[:, "RoofStyle"].map(lambda v: 1 if v=='Hip' else v)
    trainDF.loc[:, "RoofStyle"] = trainDF.loc[:, "RoofStyle"].map(lambda v: 2 if v=='Gambrel' else v)
    trainDF.loc[:, "RoofStyle"] = trainDF.loc[:, "RoofStyle"].map(lambda v: 3 if v=='Mansard' else v)
    trainDF.loc[:, "RoofStyle"] = trainDF.loc[:, "RoofStyle"].map(lambda v: 4 if v=='Flat' else v)
    trainDF.loc[:, "RoofStyle"] = trainDF.loc[:, "RoofStyle"].map(lambda v: 5 if v=='Shed' else v)
    
    testDF.loc[:, "RoofStyle"] = testDF.loc[:, "RoofStyle"].map(lambda v: 0 if v=='Gable' else v)
    testDF.loc[:, "RoofStyle"] = testDF.loc[:, "RoofStyle"].map(lambda v: 1 if v=='Hip' else v)
    testDF.loc[:, "RoofStyle"] = testDF.loc[:, "RoofStyle"].map(lambda v: 2 if v=='Gambrel' else v)
    testDF.loc[:, "RoofStyle"] = testDF.loc[:, "RoofStyle"].map(lambda v: 3 if v=='Mansard' else v)
    testDF.loc[:, "RoofStyle"] = testDF.loc[:, "RoofStyle"].map(lambda v: 4 if v=='Flat' else v)
    testDF.loc[:, "RoofStyle"] = testDF.loc[:, "RoofStyle"].map(lambda v: 5 if v=='Shed' else v)
    
    #RoofMatl
    trainDF.loc[:, "RoofMatl"] = trainDF.loc[:, "RoofMatl"].map(lambda v: 0 if v=='CompShg' else v)
    trainDF.loc[:, "RoofMatl"] = trainDF.loc[:, "RoofMatl"].map(lambda v: 1 if v=='WdShngl' else v)
    trainDF.loc[:, "RoofMatl"] = trainDF.loc[:, "RoofMatl"].map(lambda v: 2 if v=='Metal' else v)
    trainDF.loc[:, "RoofMatl"] = trainDF.loc[:, "RoofMatl"].map(lambda v: 3 if v=='WdShake' else v)
    trainDF.loc[:, "RoofMatl"] = trainDF.loc[:, "RoofMatl"].map(lambda v: 4 if v=='Membran' else v)
    trainDF.loc[:, "RoofMatl"] = trainDF.loc[:, "RoofMatl"].map(lambda v: 5 if v=='Tar&Grv' else v)
    trainDF.loc[:, "RoofMatl"] = trainDF.loc[:, "RoofMatl"].map(lambda v: 6 if v=='Roll' else v)
    trainDF.loc[:, "RoofMatl"] = trainDF.loc[:, "RoofMatl"].map(lambda v: 7 if v=='ClyTile' else v)
    
    testDF.loc[:, "RoofMatl"] = testDF.loc[:, "RoofMatl"].map(lambda v: 0 if v=='CompShg' else v)
    testDF.loc[:, "RoofMatl"] = testDF.loc[:, "RoofMatl"].map(lambda v: 1 if v=='WdShngl' else v)
    testDF.loc[:, "RoofMatl"] = testDF.loc[:, "RoofMatl"].map(lambda v: 2 if v=='Metal' else v)
    testDF.loc[:, "RoofMatl"] = testDF.loc[:, "RoofMatl"].map(lambda v: 3 if v=='WdShake' else v)
    testDF.loc[:, "RoofMatl"] = testDF.loc[:, "RoofMatl"].map(lambda v: 4 if v=='Membran' else v)
    testDF.loc[:, "RoofMatl"] = testDF.loc[:, "RoofMatl"].map(lambda v: 5 if v=='Tar&Grv' else v)
    testDF.loc[:, "RoofMatl"] = testDF.loc[:, "RoofMatl"].map(lambda v: 6 if v=='Roll' else v)
    testDF.loc[:, "RoofMatl"] = testDF.loc[:, "RoofMatl"].map(lambda v: 7 if v=='ClyTile' else v)
    
    #Exterior1st
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 0 if v=='VinylSd' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 1 if v=='MetalSd' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 2 if v=='Wd Sdng' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 3 if v=='HdBoard' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 4 if v=='BrkFace' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 5 if v=='WdShing' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 6 if v=='CemntBd' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 7 if v=='Plywood' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 8 if v=='AsbShng' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 9 if v=='Stucco' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 10 if v=='BrkComm' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 11 if v=='AsphShn' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 12 if v=='Stone' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 13 if v=='ImStucc' else v)
    trainDF.loc[:, "Exterior1st"] = trainDF.loc[:, "Exterior1st"].map(lambda v: 14 if v=='CBlock' else v)
    
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 0 if v=='VinylSd' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 1 if v=='MetalSd' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 2 if v=='Wd Sdng' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 3 if v=='HdBoard' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 4 if v=='BrkFace' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 5 if v=='WdShing' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 6 if v=='CemntBd' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 7 if v=='Plywood' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 8 if v=='AsbShng' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 9 if v=='Stucco' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 10 if v=='BrkComm' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 11 if v=='AsphShn' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 12 if v=='Stone' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 13 if v=='ImStucc' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].map(lambda v: 14 if v=='CBlock' else v)
    testDF.loc[:, "Exterior1st"] = testDF.loc[:, "Exterior1st"].fillna(15)
    
    #Exterior2nd
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 0 if v=='VinylSd' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 1 if v=='MetalSd' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 2 if v=='Wd Shng' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 3 if v=='HdBoard' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 4 if v=='Plywood' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 5 if v=='Wd Sdng' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 6 if v=='CmentBd' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 7 if v=='BrkFace' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 8 if v=='Stucco' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 9 if v=='AsbShng' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 10 if v=='Brk Cmn' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 11 if v=='ImStucc' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 12 if v=='AsphShn' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 13 if v=='Stone' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 14 if v=='Other' else v)
    trainDF.loc[:, "Exterior2nd"] = trainDF.loc[:, "Exterior2nd"].map(lambda v: 15 if v=='CBlock' else v)
    
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 0 if v=='VinylSd' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 1 if v=='MetalSd' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 2 if v=='Wd Shng' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 3 if v=='HdBoard' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 4 if v=='Plywood' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 5 if v=='Wd Sdng' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 6 if v=='CmentBd' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 7 if v=='BrkFace' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 8 if v=='Stucco' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 9 if v=='AsbShng' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 10 if v=='Brk Cmn' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 11 if v=='ImStucc' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 12 if v=='AsphShn' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 13 if v=='Stone' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 14 if v=='Other' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].map(lambda v: 15 if v=='CBlock' else v)
    testDF.loc[:, "Exterior2nd"] = testDF.loc[:, "Exterior2nd"].fillna(15)
    
    #MasVnrType
    trainDF.loc[:, "MasVnrType"] = trainDF.loc[:, "MasVnrType"].map(lambda v: 0 if v=='BrkFace' else v)
    trainDF.loc[:, "MasVnrType"] = trainDF.loc[:, "MasVnrType"].map(lambda v: 1 if v=='None' else v)
    trainDF.loc[:, "MasVnrType"] = trainDF.loc[:, "MasVnrType"].map(lambda v: 2 if v=='Stone' else v)
    trainDF.loc[:, "MasVnrType"] = trainDF.loc[:, "MasVnrType"].map(lambda v: 3 if v=='BrkCmn' else v)
    trainDF.loc[:, "MasVnrType"] = trainDF.loc[:, "MasVnrType"].fillna(4)
    
    testDF.loc[:, "MasVnrType"] = testDF.loc[:, "MasVnrType"].map(lambda v: 0 if v=='BrkFace' else v)
    testDF.loc[:, "MasVnrType"] = testDF.loc[:, "MasVnrType"].map(lambda v: 1 if v=='None' else v)
    testDF.loc[:, "MasVnrType"] = testDF.loc[:, "MasVnrType"].map(lambda v: 2 if v=='Stone' else v)
    testDF.loc[:, "MasVnrType"] = testDF.loc[:, "MasVnrType"].map(lambda v: 3 if v=='BrkCmn' else v)
    testDF.loc[:, "MasVnrType"] = testDF.loc[:, "MasVnrType"].fillna(4)
    
    #ExterQual
    trainDF.loc[:, "ExterQual"] = trainDF.loc[:, "ExterQual"].map(lambda v: 0 if v=='Gd' else v)
    trainDF.loc[:, "ExterQual"] = trainDF.loc[:, "ExterQual"].map(lambda v: 1 if v=='TA' else v)
    trainDF.loc[:, "ExterQual"] = trainDF.loc[:, "ExterQual"].map(lambda v: 2 if v=='Ex' else v)
    trainDF.loc[:, "ExterQual"] = trainDF.loc[:, "ExterQual"].map(lambda v: 3 if v=='Fa' else v)
    
    testDF.loc[:, "ExterQual"] = testDF.loc[:, "ExterQual"].map(lambda v: 0 if v=='Gd' else v)
    testDF.loc[:, "ExterQual"] = testDF.loc[:, "ExterQual"].map(lambda v: 1 if v=='TA' else v)
    testDF.loc[:, "ExterQual"] = testDF.loc[:, "ExterQual"].map(lambda v: 2 if v=='Ex' else v)
    testDF.loc[:, "ExterQual"] = testDF.loc[:, "ExterQual"].map(lambda v: 3 if v=='Fa' else v)
    
    #ExterCond
    trainDF.loc[:, "ExterCond"] = trainDF.loc[:, "ExterCond"].map(lambda v: 0 if v=='TA' else v)
    trainDF.loc[:, "ExterCond"] = trainDF.loc[:, "ExterCond"].map(lambda v: 1 if v=='Gd' else v)
    trainDF.loc[:, "ExterCond"] = trainDF.loc[:, "ExterCond"].map(lambda v: 2 if v=='Fa' else v)
    trainDF.loc[:, "ExterCond"] = trainDF.loc[:, "ExterCond"].map(lambda v: 3 if v=='Po' else v)
    trainDF.loc[:, "ExterCond"] = trainDF.loc[:, "ExterCond"].map(lambda v: 4 if v=='Ex' else v)
    
    testDF.loc[:, "ExterCond"] = testDF.loc[:, "ExterCond"].map(lambda v: 0 if v=='TA' else v)
    testDF.loc[:, "ExterCond"] = testDF.loc[:, "ExterCond"].map(lambda v: 1 if v=='Gd' else v)
    testDF.loc[:, "ExterCond"] = testDF.loc[:, "ExterCond"].map(lambda v: 2 if v=='Fa' else v)
    testDF.loc[:, "ExterCond"] = testDF.loc[:, "ExterCond"].map(lambda v: 3 if v=='Po' else v)
    testDF.loc[:, "ExterCond"] = testDF.loc[:, "ExterCond"].map(lambda v: 4 if v=='Ex' else v)
    
    #Foundation
    trainDF.loc[:, "Foundation"] = trainDF.loc[:, "Foundation"].map(lambda v: 0 if v=='PConc' else v)
    trainDF.loc[:, "Foundation"] = trainDF.loc[:, "Foundation"].map(lambda v: 1 if v=='CBlock' else v)
    trainDF.loc[:, "Foundation"] = trainDF.loc[:, "Foundation"].map(lambda v: 2 if v=='BrkTil' else v)
    trainDF.loc[:, "Foundation"] = trainDF.loc[:, "Foundation"].map(lambda v: 3 if v=='Wood' else v)
    trainDF.loc[:, "Foundation"] = trainDF.loc[:, "Foundation"].map(lambda v: 4 if v=='Slab' else v)
    trainDF.loc[:, "Foundation"] = trainDF.loc[:, "Foundation"].map(lambda v: 5 if v=='Stone' else v)
    
    testDF.loc[:, "Foundation"] = testDF.loc[:, "Foundation"].map(lambda v: 0 if v=='PConc' else v)
    testDF.loc[:, "Foundation"] = testDF.loc[:, "Foundation"].map(lambda v: 1 if v=='CBlock' else v)
    testDF.loc[:, "Foundation"] = testDF.loc[:, "Foundation"].map(lambda v: 2 if v=='BrkTil' else v)
    testDF.loc[:, "Foundation"] = testDF.loc[:, "Foundation"].map(lambda v: 3 if v=='Wood' else v)
    testDF.loc[:, "Foundation"] = testDF.loc[:, "Foundation"].map(lambda v: 4 if v=='Slab' else v)
    testDF.loc[:, "Foundation"] = testDF.loc[:, "Foundation"].map(lambda v: 5 if v=='Stone' else v)
    
    #BsmtQual
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 0 if v=='Gd' else v)
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 1 if v=='TA' else v)
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 2 if v=='Ex' else v)
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].map(lambda v: 3 if v=='Fa' else v)
    trainDF.loc[:, "BsmtQual"] = trainDF.loc[:, "BsmtQual"].fillna(4)
    
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].map(lambda v: 0 if v=='Gd' else v)
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].map(lambda v: 1 if v=='TA' else v)
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].map(lambda v: 2 if v=='Ex' else v)
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].map(lambda v: 3 if v=='Fa' else v)
    testDF.loc[:, "BsmtQual"] = testDF.loc[:, "BsmtQual"].fillna(4)
    
    #BsmtCond
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 0 if v=='Gd' else v)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 1 if v=='TA' else v)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 2 if v=='Po' else v)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].map(lambda v: 3 if v=='Fa' else v)
    trainDF.loc[:, "BsmtCond"] = trainDF.loc[:, "BsmtCond"].fillna(4)
    
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].map(lambda v: 0 if v=='Gd' else v)
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].map(lambda v: 1 if v=='TA' else v)
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].map(lambda v: 2 if v=='Po' else v)
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].map(lambda v: 3 if v=='Fa' else v)
    testDF.loc[:, "BsmtCond"] = testDF.loc[:, "BsmtCond"].fillna(4)
    
    #BsmtExposure
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 0 if v=='No' else v)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 1 if v=='Gd' else v)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 2 if v=='Mn' else v)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].map(lambda v: 3 if v=='Av' else v)
    trainDF.loc[:, "BsmtExposure"] = trainDF.loc[:, "BsmtExposure"].fillna(4)
    
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].map(lambda v: 0 if v=='No' else v)
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].map(lambda v: 1 if v=='Gd' else v)
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].map(lambda v: 2 if v=='Mn' else v)
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].map(lambda v: 3 if v=='Av' else v)
    testDF.loc[:, "BsmtExposure"] = testDF.loc[:, "BsmtExposure"].fillna(4)
 
    #BsmtFinType1
    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].map(lambda v: 0 if v=='GLQ' else v)
    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].map(lambda v: 1 if v=='ALQ' else v)
    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].map(lambda v: 2 if v=='Unf' else v)
    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].map(lambda v: 3 if v=='Rec' else v)
    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].map(lambda v: 4 if v=='BLQ' else v)
    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].map(lambda v: 5 if v=='LwQ' else v)
    trainDF.loc[:, "BsmtFinType1"] = trainDF.loc[:, "BsmtFinType1"].fillna(6)
    
    testDF.loc[:, "BsmtFinType1"] = testDF.loc[:, "BsmtFinType1"].map(lambda v: 0 if v=='GLQ' else v)
    testDF.loc[:, "BsmtFinType1"] = testDF.loc[:, "BsmtFinType1"].map(lambda v: 1 if v=='ALQ' else v)
    testDF.loc[:, "BsmtFinType1"] = testDF.loc[:, "BsmtFinType1"].map(lambda v: 2 if v=='Unf' else v)
    testDF.loc[:, "BsmtFinType1"] = testDF.loc[:, "BsmtFinType1"].map(lambda v: 3 if v=='Rec' else v)
    testDF.loc[:, "BsmtFinType1"] = testDF.loc[:, "BsmtFinType1"].map(lambda v: 4 if v=='BLQ' else v)
    testDF.loc[:, "BsmtFinType1"] = testDF.loc[:, "BsmtFinType1"].map(lambda v: 5 if v=='LwQ' else v)
    testDF.loc[:, "BsmtFinType1"] = testDF.loc[:, "BsmtFinType1"].fillna(6)
    
    #BsmtFinType2
    trainDF.loc[:, "BsmtFinType2"] = trainDF.loc[:, "BsmtFinType2"].map(lambda v: 0 if v=='Unf' else v)
    trainDF.loc[:, "BsmtFinType2"] = trainDF.loc[:, "BsmtFinType2"].map(lambda v: 1 if v=='BLQ' else v)
    trainDF.loc[:, "BsmtFinType2"] = trainDF.loc[:, "BsmtFinType2"].map(lambda v: 2 if v=='ALQ' else v)
    trainDF.loc[:, "BsmtFinType2"] = trainDF.loc[:, "BsmtFinType2"].map(lambda v: 3 if v=='Rec' else v)
    trainDF.loc[:, "BsmtFinType2"] = trainDF.loc[:, "BsmtFinType2"].map(lambda v: 4 if v=='LwQ' else v)
    trainDF.loc[:, "BsmtFinType2"] = trainDF.loc[:, "BsmtFinType2"].map(lambda v: 5 if v=='GLQ' else v)
    trainDF.loc[:, "BsmtFinType2"] = trainDF.loc[:, "BsmtFinType2"].fillna(6)
    
    testDF.loc[:, "BsmtFinType2"] = testDF.loc[:, "BsmtFinType2"].map(lambda v: 0 if v=='Unf' else v)
    testDF.loc[:, "BsmtFinType2"] = testDF.loc[:, "BsmtFinType2"].map(lambda v: 1 if v=='BLQ' else v)
    testDF.loc[:, "BsmtFinType2"] = testDF.loc[:, "BsmtFinType2"].map(lambda v: 2 if v=='ALQ' else v)
    testDF.loc[:, "BsmtFinType2"] = testDF.loc[:, "BsmtFinType2"].map(lambda v: 3 if v=='Rec' else v)
    testDF.loc[:, "BsmtFinType2"] = testDF.loc[:, "BsmtFinType2"].map(lambda v: 4 if v=='LwQ' else v)
    testDF.loc[:, "BsmtFinType2"] = testDF.loc[:, "BsmtFinType2"].map(lambda v: 5 if v=='GLQ' else v)
    testDF.loc[:, "BsmtFinType2"] = testDF.loc[:, "BsmtFinType2"].fillna(6)
    
    #Heating
    trainDF.loc[:, "Heating"] = trainDF.loc[:, "Heating"].map(lambda v: 0 if v=='GasA' else v)
    trainDF.loc[:, "Heating"] = trainDF.loc[:, "Heating"].map(lambda v: 1 if v=='GasW' else v)
    trainDF.loc[:, "Heating"] = trainDF.loc[:, "Heating"].map(lambda v: 2 if v=='Grav' else v)
    trainDF.loc[:, "Heating"] = trainDF.loc[:, "Heating"].map(lambda v: 3 if v=='Wall' else v)
    trainDF.loc[:, "Heating"] = trainDF.loc[:, "Heating"].map(lambda v: 4 if v=='OthW' else v)
    trainDF.loc[:, "Heating"] = trainDF.loc[:, "Heating"].map(lambda v: 5 if v=='Floor' else v)
    
    testDF.loc[:, "Heating"] = testDF.loc[:, "Heating"].map(lambda v: 0 if v=='GasA' else v)
    testDF.loc[:, "Heating"] = testDF.loc[:, "Heating"].map(lambda v: 1 if v=='GasW' else v)
    testDF.loc[:, "Heating"] = testDF.loc[:, "Heating"].map(lambda v: 2 if v=='Grav' else v)
    testDF.loc[:, "Heating"] = testDF.loc[:, "Heating"].map(lambda v: 3 if v=='Wall' else v)
    testDF.loc[:, "Heating"] = testDF.loc[:, "Heating"].map(lambda v: 4 if v=='OthW' else v)
    testDF.loc[:, "Heating"] = testDF.loc[:, "Heating"].map(lambda v: 5 if v=='Floor' else v)
    
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
    
    #CentralAir
    trainDF.loc[:, "CentralAir"] = trainDF.loc[:, "CentralAir"].map(lambda v: 0 if v=='Y' else v)
    trainDF.loc[:, "CentralAir"] = trainDF.loc[:, "CentralAir"].map(lambda v: 1 if v=='N' else v)
    
    testDF.loc[:, "CentralAir"] = testDF.loc[:, "CentralAir"].map(lambda v: 0 if v=='Y' else v)
    testDF.loc[:, "CentralAir"] = testDF.loc[:, "CentralAir"].map(lambda v: 1 if v=='N' else v)
    
    #Electrical
    trainDF.loc[:, "Electrical"] = trainDF.loc[:, "Electrical"].map(lambda v: 0 if v=='SBrkr' else v)
    trainDF.loc[:, "Electrical"] = trainDF.loc[:, "Electrical"].map(lambda v: 1 if v=='FuseF' else v)
    trainDF.loc[:, "Electrical"] = trainDF.loc[:, "Electrical"].map(lambda v: 2 if v=='FuseA' else v)
    trainDF.loc[:, "Electrical"] = trainDF.loc[:, "Electrical"].map(lambda v: 3 if v=='FuseP' else v)
    trainDF.loc[:, "Electrical"] = trainDF.loc[:, "Electrical"].map(lambda v: 4 if v=='Mix' else v)
    trainDF.loc[:, "Electrical"] = trainDF.loc[:, "Electrical"].fillna(5)
    
    testDF.loc[:, "Electrical"] = testDF.loc[:, "Electrical"].map(lambda v: 0 if v=='SBrkr' else v)
    testDF.loc[:, "Electrical"] = testDF.loc[:, "Electrical"].map(lambda v: 1 if v=='FuseF' else v)
    testDF.loc[:, "Electrical"] = testDF.loc[:, "Electrical"].map(lambda v: 2 if v=='FuseA' else v)
    testDF.loc[:, "Electrical"] = testDF.loc[:, "Electrical"].map(lambda v: 3 if v=='FuseP' else v)
    testDF.loc[:, "Electrical"] = testDF.loc[:, "Electrical"].map(lambda v: 4 if v=='Mix' else v)
    testDF.loc[:, "Electrical"] = testDF.loc[:, "Electrical"].fillna(5)
    
    #KitchenQual
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 0 if v=='Gd' else v)
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 1 if v=='TA' else v)
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 2 if v=='Ex' else v)
    trainDF.loc[:, "KitchenQual"] = trainDF.loc[:, "KitchenQual"].map(lambda v: 3 if v=='Fa' else v)
    
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].map(lambda v: 0 if v=='Gd' else v)
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].map(lambda v: 1 if v=='TA' else v)
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].map(lambda v: 2 if v=='Ex' else v)
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].map(lambda v: 3 if v=='Fa' else v)
    testDF.loc[:, "KitchenQual"] = testDF.loc[:, "KitchenQual"].fillna(4)
    
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
    testDF.loc[:, "Functional"] = testDF.loc[:, "Functional"].fillna(7)
    
    #FireplaceQu
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 0 if v=='TA' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 1 if v=='Gd' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 2 if v=='Fa' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 3 if v=='Ex' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].map(lambda v: 4 if v=='Po' else v)
    trainDF.loc[:, "FireplaceQu"] = trainDF.loc[:, "FireplaceQu"].fillna(5)
    
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 0 if v=='TA' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 1 if v=='Gd' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 2 if v=='Fa' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 3 if v=='Ex' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].map(lambda v: 4 if v=='Po' else v)
    testDF.loc[:, "FireplaceQu"] = testDF.loc[:, "FireplaceQu"].fillna(5)
    
    #GarageType
    trainDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].map(lambda v: 0 if v=='Attchd' else v)
    trainDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].map(lambda v: 1 if v=='Detchd' else v)
    trainDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].map(lambda v: 2 if v=='BuiltIn' else v)
    trainDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].map(lambda v: 3 if v=='CarPort' else v)
    trainDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].map(lambda v: 4 if v=='Basment' else v)
    trainDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].map(lambda v: 5 if v=='2Types' else v)
    trainDF.loc[:, "GarageType"] = trainDF.loc[:, "GarageType"].fillna(6)
    
    testDF.loc[:, "GarageType"] = testDF.loc[:, "GarageType"].map(lambda v: 0 if v=='Attchd' else v)
    testDF.loc[:, "GarageType"] = testDF.loc[:, "GarageType"].map(lambda v: 1 if v=='Detchd' else v)
    testDF.loc[:, "GarageType"] = testDF.loc[:, "GarageType"].map(lambda v: 2 if v=='BuiltIn' else v)
    testDF.loc[:, "GarageType"] = testDF.loc[:, "GarageType"].map(lambda v: 3 if v=='CarPort' else v)
    testDF.loc[:, "GarageType"] = testDF.loc[:, "GarageType"].map(lambda v: 4 if v=='Basment' else v)
    testDF.loc[:, "GarageType"] = testDF.loc[:, "GarageType"].map(lambda v: 5 if v=='2Types' else v)
    testDF.loc[:, "GarageType"] = testDF.loc[:, "GarageType"].fillna(6)
    
    #GarageFinish
    trainDF.loc[:, "GarageFinish"] = trainDF.loc[:, "GarageFinish"].map(lambda v: 0 if v=='RFn' else v)
    trainDF.loc[:, "GarageFinish"] = trainDF.loc[:, "GarageFinish"].map(lambda v: 1 if v=='Unf' else v)
    trainDF.loc[:, "GarageFinish"] = trainDF.loc[:, "GarageFinish"].map(lambda v: 2 if v=='Fin' else v)
    trainDF.loc[:, "GarageFinish"] = trainDF.loc[:, "GarageFinish"].fillna(4)
    
    testDF.loc[:, "GarageFinish"] = testDF.loc[:, "GarageFinish"].map(lambda v: 0 if v=='RFn' else v)
    testDF.loc[:, "GarageFinish"] = testDF.loc[:, "GarageFinish"].map(lambda v: 1 if v=='Unf' else v)
    testDF.loc[:, "GarageFinish"] = testDF.loc[:, "GarageFinish"].map(lambda v: 2 if v=='Fin' else v)
    testDF.loc[:, "GarageFinish"] = testDF.loc[:, "GarageFinish"].fillna(4)
    
    #GarageQual
    trainDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].map(lambda v: 0 if v=='TA' else v)
    trainDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].map(lambda v: 1 if v=='Fa' else v)
    trainDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].map(lambda v: 2 if v=='Gd' else v)
    trainDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].map(lambda v: 3 if v=='Ex' else v)
    trainDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].map(lambda v: 4 if v=='Po' else v)
    trainDF.loc[:, "GarageQual"] = trainDF.loc[:, "GarageQual"].fillna(5)
    
    testDF.loc[:, "GarageQual"] = testDF.loc[:, "GarageQual"].map(lambda v: 0 if v=='TA' else v)
    testDF.loc[:, "GarageQual"] = testDF.loc[:, "GarageQual"].map(lambda v: 1 if v=='Fa' else v)
    testDF.loc[:, "GarageQual"] = testDF.loc[:, "GarageQual"].map(lambda v: 2 if v=='Gd' else v)
    testDF.loc[:, "GarageQual"] = testDF.loc[:, "GarageQual"].map(lambda v: 3 if v=='Ex' else v)
    testDF.loc[:, "GarageQual"] = testDF.loc[:, "GarageQual"].map(lambda v: 4 if v=='Po' else v)
    testDF.loc[:, "GarageQual"] = testDF.loc[:, "GarageQual"].fillna(5)
    
    #GarageCond
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 0 if v=='TA' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 1 if v=='Fa' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 2 if v=='Gd' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 3 if v=='Ex' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].map(lambda v: 4 if v=='Po' else v)
    trainDF.loc[:, "GarageCond"] = trainDF.loc[:, "GarageCond"].fillna(5)
    
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 0 if v=='TA' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 1 if v=='Fa' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 2 if v=='Gd' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 3 if v=='Ex' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].map(lambda v: 4 if v=='Po' else v)
    testDF.loc[:, "GarageCond"] = testDF.loc[:, "GarageCond"].fillna(5)
    
    #PavedDrive
    trainDF.loc[:, "PavedDrive"] = trainDF.loc[:, "PavedDrive"].map(lambda v: 0 if v=='Y' else v)
    trainDF.loc[:, "PavedDrive"] = trainDF.loc[:, "PavedDrive"].map(lambda v: 1 if v=='N' else v)
    trainDF.loc[:, "PavedDrive"] = trainDF.loc[:, "PavedDrive"].map(lambda v: 2 if v=='P' else v)
    
    testDF.loc[:, "PavedDrive"] = testDF.loc[:, "PavedDrive"].map(lambda v: 0 if v=='Y' else v)
    testDF.loc[:, "PavedDrive"] = testDF.loc[:, "PavedDrive"].map(lambda v: 1 if v=='N' else v)
    testDF.loc[:, "PavedDrive"] = testDF.loc[:, "PavedDrive"].map(lambda v: 2 if v=='P' else v)
    
    #PoolQC
    trainDF.loc[:, "PoolQC"] = trainDF.loc[:, "PoolQC"].map(lambda v: 0 if v=='Ex' else v)
    trainDF.loc[:, "PoolQC"] = trainDF.loc[:, "PoolQC"].map(lambda v: 1 if v=='Fa' else v)
    trainDF.loc[:, "PoolQC"] = trainDF.loc[:, "PoolQC"].map(lambda v: 2 if v=='Gd' else v)
    trainDF.loc[:, "PoolQC"] = trainDF.loc[:, "PoolQC"].fillna(3)
    
    testDF.loc[:, "PoolQC"] = testDF.loc[:, "PoolQC"].map(lambda v: 0 if v=='Ex' else v)
    testDF.loc[:, "PoolQC"] = testDF.loc[:, "PoolQC"].map(lambda v: 1 if v=='Fa' else v)
    testDF.loc[:, "PoolQC"] = testDF.loc[:, "PoolQC"].map(lambda v: 2 if v=='Gd' else v)
    testDF.loc[:, "PoolQC"] = testDF.loc[:, "PoolQC"].fillna(3)
    
    #Fence 
    trainDF.loc[:, "Fence"] = trainDF.loc[:, "Fence"].map(lambda v: 0 if v=='MnPrv' else v)
    trainDF.loc[:, "Fence"] = trainDF.loc[:, "Fence"].map(lambda v: 1 if v=='GdWo' else v)
    trainDF.loc[:, "Fence"] = trainDF.loc[:, "Fence"].map(lambda v: 2 if v=='GdPrv' else v)
    trainDF.loc[:, "Fence"] = trainDF.loc[:, "Fence"].map(lambda v: 3 if v=='MnWw' else v)
    trainDF.loc[:, "Fence"] = trainDF.loc[:, "Fence"].fillna(4)
    
    testDF.loc[:, "Fence"] = testDF.loc[:, "Fence"].map(lambda v: 0 if v=='MnPrv' else v)
    testDF.loc[:, "Fence"] = testDF.loc[:, "Fence"].map(lambda v: 1 if v=='GdWo' else v)
    testDF.loc[:, "Fence"] = testDF.loc[:, "Fence"].map(lambda v: 2 if v=='GdPrv' else v)
    testDF.loc[:, "Fence"] = testDF.loc[:, "Fence"].map(lambda v: 3 if v=='MnWw' else v)
    testDF.loc[:, "Fence"] = testDF.loc[:, "Fence"].fillna(4)
    
    
    #MiscFeature
    trainDF.loc[:, "MiscFeature"] = trainDF.loc[:, "MiscFeature"].map(lambda v: 0 if v=='Shed' else v)
    trainDF.loc[:, "MiscFeature"] = trainDF.loc[:, "MiscFeature"].map(lambda v: 1 if v=='Gar2' else v)
    trainDF.loc[:, "MiscFeature"] = trainDF.loc[:, "MiscFeature"].map(lambda v: 2 if v=='Othr' else v)
    trainDF.loc[:, "MiscFeature"] = trainDF.loc[:, "MiscFeature"].map(lambda v: 3 if v=='TenC' else v)
    trainDF.loc[:, "MiscFeature"] = trainDF.loc[:, "MiscFeature"].fillna(4)
    
    testDF.loc[:, "MiscFeature"] = testDF.loc[:, "MiscFeature"].map(lambda v: 0 if v=='Shed' else v)
    testDF.loc[:, "MiscFeature"] = testDF.loc[:, "MiscFeature"].map(lambda v: 1 if v=='Gar2' else v)
    testDF.loc[:, "MiscFeature"] = testDF.loc[:, "MiscFeature"].map(lambda v: 2 if v=='Othr' else v)
    testDF.loc[:, "MiscFeature"] = testDF.loc[:, "MiscFeature"].map(lambda v: 3 if v=='TenC' else v)
    testDF.loc[:, "MiscFeature"] = testDF.loc[:, "MiscFeature"].fillna(4)
    
    #SaleType
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 0 if v=='WD' else v)
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 1 if v=='New' else v)
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 2 if v=='COD' else v)
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 3 if v=='ConLD' else v)
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 4 if v=='ConLI' else v)
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 5 if v=='CWD' else v)
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 6 if v=='ConLw' else v)
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 7 if v=='Con' else v)
    trainDF.loc[:, "SaleType"] = trainDF.loc[:, "SaleType"].map(lambda v: 8 if v=='Oth' else v)
    
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 0 if v=='WD' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 1 if v=='New' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 2 if v=='COD' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 3 if v=='ConLD' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 4 if v=='ConLI' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 5 if v=='CWD' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 6 if v=='ConLw' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 7 if v=='Con' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].map(lambda v: 8 if v=='Oth' else v)
    testDF.loc[:, "SaleType"] = testDF.loc[:, "SaleType"].fillna(9)
    
    #SaleCondition
    trainDF.loc[:, "SaleCondition"] = trainDF.loc[:, "SaleCondition"].map(lambda v: 0 if v=='Normal' else v)
    trainDF.loc[:, "SaleCondition"] = trainDF.loc[:, "SaleCondition"].map(lambda v: 1 if v=='Abnorml' else v)
    trainDF.loc[:, "SaleCondition"] = trainDF.loc[:, "SaleCondition"].map(lambda v: 2 if v=='Partial' else v)
    trainDF.loc[:, "SaleCondition"] = trainDF.loc[:, "SaleCondition"].map(lambda v: 3 if v=='AdjLand' else v)
    trainDF.loc[:, "SaleCondition"] = trainDF.loc[:, "SaleCondition"].map(lambda v: 4 if v=='Alloca' else v)
    trainDF.loc[:, "SaleCondition"] = trainDF.loc[:, "SaleCondition"].map(lambda v: 5 if v=='Family' else v)
    
    testDF.loc[:, "SaleCondition"] = testDF.loc[:, "SaleCondition"].map(lambda v: 0 if v=='Normal' else v)
    testDF.loc[:, "SaleCondition"] = testDF.loc[:, "SaleCondition"].map(lambda v: 1 if v=='Abnorml' else v)
    testDF.loc[:, "SaleCondition"] = testDF.loc[:, "SaleCondition"].map(lambda v: 2 if v=='Partial' else v)
    testDF.loc[:, "SaleCondition"] = testDF.loc[:, "SaleCondition"].map(lambda v: 3 if v=='AdjLand' else v)
    testDF.loc[:, "SaleCondition"] = testDF.loc[:, "SaleCondition"].map(lambda v: 4 if v=='Alloca' else v)
    testDF.loc[:, "SaleCondition"] = testDF.loc[:, "SaleCondition"].map(lambda v: 5 if v=='Family' else v)
    
    #get_Dummies
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    '''
    print("trainInput before:", trainInput, '\n', sep='\n')
   
    trainInput = pd.get_dummies(trainInput, columns=['LotShape','Street','Alley', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2','HouseStyle', 'Exterior1st', 'Exterior2nd', 'ExterQual','ExterCond', 'Foundation','BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','HeatingQC','CentralAir','Electrical','KitchenQual', 'Functional', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageCond','PavedDrive','SaleType' ])
    testInput = pd.get_dummies(testInput, columns=['LotShape','Street','Alley', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2','HouseStyle', 'Exterior1st', 'Exterior2nd', 'ExterQual','ExterCond', 'Foundation','BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','HeatingQC','CentralAir','Electrical','KitchenQual', 'Functional', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageCond','PavedDrive','SaleType' ])
    
    print("trainInput after:", trainInput, '\n', sep='\n')
    predictors = trainInput.columns
   
    print("Predictors:", predictors)
    
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

