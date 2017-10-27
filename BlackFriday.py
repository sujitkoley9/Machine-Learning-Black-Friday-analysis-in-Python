 #----------------- Step 1:  Importing required packages for this problem ----------------------------------- 
   # data analysis and wrangling
    import pandas as pd
    import numpy as np
    import random as rn
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
   
    # machine learning
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import cross_val_score
    
    import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    from xgboost.sklearn import XGBRegressor
    from xgboost  import plot_importance
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    
    
  #--------- Step 2:  Reading and loading train and test datasets and generate data quality report----------- 
   
    # loading train and test sets with pandas 
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    #append two train  and test dataframe
    full  = train_df.append(test_df,ignore_index=True)
    
    # Print the columns of dataframe
    print(full.columns.values)
    
    # Returns first n rows
    full.head(10)
    
    
    # Retrive data type of object and no. of non-null object
    full.info()
    
    # Retrive details of integer and float data type 
    full.describe()
    
    # To get  details of the categorical types
    full.describe(include=['O'])

   

  #Prepare data quality report-
  # To get count of no. of NULL for each data type columns = full.columns.values
    columns = full.columns.values
    data_types = pd.DataFrame(full.dtypes, columns=['data types'])
    
    missing_data_counts = pd.DataFrame(full.isnull().sum(),
                            columns=['Missing Values'])
    
    present_data_counts = pd.DataFrame(full.count(), columns=['Present Values'])
    
    UniqueValues = pd.DataFrame(full.nunique(), columns=['Unique Values'])
    
    MinimumValues = pd.DataFrame(columns=['Minimum Values'])
    for c in list(columns):
       if (full[c].dtypes == 'float64' ) | (full[c].dtypes == 'int64'):
            MinimumValues.loc[c]=full[c].min()
       else:
            MinimumValues.loc[c]=0
 
    MaximumValues = pd.DataFrame(columns=['Maximum Values'])
    for c in list(columns):
       if (full[c].dtypes == 'float64' ) |(full[c].dtypes == 'int64'):
            MaximumValues.loc[c]=full[c].max()
       else:
            MaximumValues.loc[c]=0
    
    data_quality_report=data_types.join(missing_data_counts).join(present_data_counts).join(UniqueValues).join(MinimumValues).join(MaximumValues)
    data_quality_report.to_csv('Black_Friday_data_quality.csv', index=True)



   
   
   
#---------------Step 3: Missing value treatment----------------------------------------------------------------  
  
    
     # Treatment for Product_Category_2
   full['Product_Category_2'].fillna(full['Product_Category_2'].mean(), inplace=True)
 
  
     # Treatment for Product_Category_3
  
   full['Product_Category_3'].fillna(full['Product_Category_3'].mean(), inplace=True)
   
 
#--------------Step 4:Outlier Treatment ----------------------------------------------------------------------  

#  outlier treatment using BoxPlot 
   
    # Product_Category_1
     BoxPlot=boxplot(full['Product_Category_1'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     full.loc[full['Product_Category_1'].isin(outlier),'Product_Category_1']=full['Product_Category_1'].mean()
     
     #Product_Category_2
     BoxPlot=boxplot(full['Product_Category_2'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     # There is no outlier, No need any operation
     #full.loc[full['Product_Category_2'].isin(outlier),'Product_Category_2']=full['Product_Category_2'].mean()
     
      
     #Product_Category_3
     BoxPlot=boxplot(full['Product_Category_3'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     full.loc[full['Product_Category_3'].isin(outlier),'Product_Category_3']=full['Product_Category_3'].mean()

     #Purchase
     BoxPlot=boxplot(full[0:550068]['Purchase'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     full.loc[full['Purchase'].isin(outlier),'Purchase']=full[0:550068]['Purchase'].mean()
    

#-----------------Step 5:Exploration analysis of data---------------------------------------------------------

       # Create photocopy of  trian portion of full and assign it full1
        full1=full[0:550068].copy()
        
       # Analying relation between Age & Purchase
        Full_analysis=full1[['Age','Purchase']].groupby('Age',as_index=False).mean().sort_values(by='Purchase', ascending=False)
       
        g= sns.factorplot(x='Age', 
                          y='Purchase',
                          data=Full_analysis, 
                          hue='Age',
                          kind='bar') # barplot
        
         # Analying relation between City_Category & Purchase
        Full_analysis=full1[['City_Category','Purchase']].groupby('City_Category',as_index=False).mean().sort_values(by='Purchase', ascending=False)
       
        g= sns.factorplot(x='City_Category', 
                          y='Purchase',
                          data=Full_analysis, 
                          hue='City_Category',
                          kind='bar') # barplot
        
          # Analying relation between Gender & Purchase
        Full_analysis=full1[['Gender','Purchase']].groupby('Gender',as_index=False).mean().sort_values(by='Purchase', ascending=False)
       
        g= sns.factorplot(x='Gender', 
                          y='Purchase',
                          data=Full_analysis, 
                          hue='Gender',
                          kind='bar') # barplot 
        
        
         # Analying relation between Stay_In_Current_City_Years & Purchase
        Full_analysis=full1[['Stay_In_Current_City_Years','Purchase']].groupby('Stay_In_Current_City_Years',as_index=False).mean().sort_values(by='Purchase', ascending=False)
       
        g= sns.factorplot(x='Stay_In_Current_City_Years', 
                          y='Purchase',
                          data=Full_analysis, 
                          hue='Stay_In_Current_City_Years',
                          kind='bar') # barplot 
        
        
         # Analying relation between Marital_Status & Purchase
        Full_analysis=full1[['Marital_Status','Purchase']].groupby('Marital_Status',as_index=False).mean().sort_values(by='Purchase', ascending=False)
       
        g= sns.factorplot(x='Marital_Status', 
                          y='Purchase',
                          data=Full_analysis, 
                          hue='Marital_Status',
                          kind='bar') # barplot 
        
         # Analying relation between Product_Category_1 & Purchase
        sns.lmplot(x='Product_Category_1', y='Purchase', data=full1)
        
         # Analying relation between Product_Category_2 & Purchase
        sns.lmplot(x='Product_Category_2', y='Purchase', data=full1)
        
         # Analying relation between Product_Category_3 & Purchase
        sns.lmplot(x='Product_Category_3', y='Purchase', data=full1)
        
       # Analying relation between Occupation & Purchase
        sns.lmplot(x='Occupation', y='Purchase', data=full1)
        
     
   
     
       
     
       
       

#------------------------------------Step 6:Feature Engineering--------------------------------------
  
   
   # Create dummy variable for Age
   Age_dummies = pd.get_dummies(full['Age'],prefix='Age')
   Age_dummies=Age_dummies.iloc[:,1:]
   full=full.join(Age_dummies)
   
   
   #Creating dummy variable for City_Category
   City_Category_dummies = pd.get_dummies(full['City_Category'],prefix='City_Category')
   City_Category_dummies=City_Category_dummies.iloc[:,1:]
   full=full.join(City_Category_dummies)
   
    #Creating dummy variable for Gender
   Gender_dummies = pd.get_dummies(full['Gender'],prefix='Gender')
   Gender_dummies=Gender_dummies.iloc[:,1:]
   full=full.join(Gender_dummies)
   
    #Creating dummy variable for Stay_In_Current_City_Years
   Stay_In_Current_City_Years_dummies = pd.get_dummies(full['Stay_In_Current_City_Years'],prefix='Stay_In_Current_City_Years')
   Stay_In_Current_City_Years_dummies=Stay_In_Current_City_Years_dummies.iloc[:,1:]
   full=full.join(Stay_In_Current_City_Years_dummies)
   
   
   
   
   
     #---------------------------------- Droping unnecessary columns-------------------------------
    full.drop(['User_ID','Product_ID','Age','City_Category',
               'Gender','Stay_In_Current_City_Years'], axis=1, inplace=True)
    
    full.columns.values

   
   
   
#----------------------Step 7: Separating train/test dataset and Normalize data--------------------------------   
    train_new=full[0:550068]
    test_new=full[550068:]
    
    X_train = train_new.drop(['Purchase'], axis=1)
    Y_train = train_new["Purchase"]
    
    X_test  = test_new.drop(['Purchase'], axis=1)
   
    #-----Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
     #--------------------PCA to reduce dimension and remove correlation----------------------------    
     pca = PCA(n_components =18)
     pca.fit_transform(X_train)
     #The amount of variance that each PC explains
     var= pca.explained_variance_ratio_
     #Cumulative Variance explains
     var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
     plt.plot(var1)
     
     # As per analysis, we can skip 4 principal componet, use only 11 components
     
     pca = PCA(n_components =16)
     X_train=pca.fit_transform(X_train)
     X_test=pca.fit_transform(X_test)
     
     
#----------------------Step 8:Run Algorithm----------------------------------------------------------------------
  #1.Logistic Regression
    
    Linreg = LinearRegression()
    Linreg.fit(X_train, Y_train)
    Linreg_score = cross_val_score(estimator = Linreg, X = X_train, y = Y_train, cv =    10,
                                 scoring='neg_mean_squared_error')
    
    Linreg_score = (np.sqrt(np.abs(Linreg_score)))
    #RMSE
    Linreg_score_mean = Linreg_score.mean()
    Linreg_score_std = Linreg_score.std()
    
  #2.Decision Tree
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train, Y_train)
    decision_tree_score = cross_val_score(estimator = decision_tree, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    
    decision_tree_score = (np.sqrt(np.abs(decision_tree_score)))
    #RMSE
    decision_tree_score_mean = decision_tree_score.mean()
    decision_tree_score_std = decision_tree_score.std()
    
    # Choose some parameter combinations to try
    parameters = {
                  'max_features': ['log2', 'sqrt','auto'],
                  'criterion': ['mse', 'friedman_mse'],
                  'max_depth': range(2,10), 
                  'min_samples_split': range(2,10),
                  'min_samples_leaf': range(1,10)
                 }

    # Search for best parameters
    grid_obj = GridSearchCV(estimator=decision_tree, 
                                    param_grid= parameters,
                                    scoring = 'neg_mean_squared_error',
                                    cv = 3,n_jobs=-1)
    

    
    grid_obj = grid_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    decision_tree_best = grid_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    decision_tree_best.fit(X_train, Y_train)
    
    # Calculate accuracy of decisison tree again
    decision_tree_score = cross_val_score(estimator = decision_tree_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    
    decision_tree_score = (np.sqrt(np.abs(decision_tree_score)))
    
    decision_tree_score_mean = decision_tree_score.mean()
    
    decision_tree_score_std = decision_tree_score.std()
    #---To Know importanve of variable
    feature_importance = pd.Series(decision_tree_best.feature_importances_, X_train.columns.values).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    
   
    
    
    
    #3.Random Forest
    random_forest = RandomForestRegressor(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    random_forest_score = cross_val_score(estimator = random_forest, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    random_forest_score = (np.sqrt(np.abs(random_forest_score)))
    
    random_forest_score_mean = random_forest_score.mean()
    random_forest_score_std  = random_forest_score.std()


    # Choose some parameter combinations to try
    parameters = { 
                 'max_features': ['log2', 'sqrt','auto'], 
                 'criterion': ['mse', 'mae'],
                 'max_depth': range(2,10), 
                 'min_samples_split': range(2,10),
                 'min_samples_leaf': range(1,10)
                 }
    
     # Choose some parameter combinations to try
    parameters = { 
                  'max_features': ['auto'], 
                  'criterion': ['mse'],
                  'max_depth': [8], 
                 'min_samples_split': [3],
                 'min_samples_leaf': [3]
                 }
   
    grid_obj = GridSearchCV(estimator=random_forest, 
                                    param_grid= parameters,
                                    scoring = 'neg_mean_squared_error',
                                    cv = 3,n_jobs=-1)
    

    
    grid_obj = grid_obj.fit(X_train, Y_train)
    

    
   

    # Set the clf to the best combination of parameters
    random_forest_best = grid_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    random_forest_best.fit(X_train, Y_train)
    random_forest_score = cross_val_score(estimator = random_forest_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    
    random_forest_score = (np.sqrt(np.abs(random_forest_score)))
    
    random_forest_score_mean = random_forest_score.mean()
    random_forest_score_std  = random_forest_score.std()
    
    #---To Know importanve of variable
    feature_importance = pd.Series(random_forest_best.feature_importances_, X_train.columns.values).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    
    #4.XGBOOST
    Xgboost = XGBRegressor()
    Xgboost.fit(X_train, Y_train)
    Xgboost_score = cross_val_score(estimator = Xgboost, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    Xgboost_score = (np.sqrt(np.abs(Xgboost_score)))
    
    Xgboost_score_mean = Xgboost_score.mean()
    Xgboost_score_std  = Xgboost_score.std()



    # Choose some parameter combinations to try
   parameters = {'learning_rate':np.arange(0.1, .5, 0.1),
                  'n_estimators':[200],
                  'max_depth': range(4,10),
                  'min_child_weight':range(1,5),
                  'reg_lambda':np.arange(0.55, .9, 0.05),
                  'subsample':np.arange(0.1, 1, 0.1),
                  'colsample_bytree':np.arange(0.1, 1, 0.1)
               }
   
     # Choose some parameter combinations to try
   parameters = {'learning_rate':[.1,.3,.02],
                  'n_estimators':[73,74,75],
                  'max_depth': [3,4],
                  'min_child_weight':[1,2],
                  'reg_lambda':np.arange(0.55, .9, 0.05),
                  'subsample':[.8,0.9,1],
                  'colsample_bytree':[.8,0.9,1]
               }
    
    # Search for best parameters
   Random_obj = RandomizedSearchCV(estimator=Xgboost, 
                                  param_distributions = parameters,
                                  scoring = 'neg_mean_squared_error',
                                  cv = 3,n_iter=300,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train) 

    # Set the clf to the best combination of parameters
    Xgboost_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    Xgboost_best.fit(X_train, Y_train)
    
    Xgboost_score = cross_val_score(estimator = Xgboost_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    Xgboost_score = (np.sqrt(np.abs(Xgboost_score)))
    
    Xgboost_score_mean = Xgboost_score.mean()
    Xgboost_score_std  = Xgboost_score.std()


    #---To Know importanve of variable
    plot_importance(Xgboost_best)
    pyplot.show()
    
 #5.SVM
    SVM_model=SVR()
    SVM_model.fit(X_train, Y_train)
    SVM_model_score = cross_val_score(estimator = SVM_model, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    SVM_model_score = (np.sqrt(np.abs(SVM_model_score)))
    
    SVM_model_score_mean = SVM_model_score.mean()
    SVM_model_score_std  = SVM_model_score.std()




    # Choose some parameter combinations to try
   parameters = { 'kernel':('linear', 'rbf'),
                  'gamma': [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5],
                  'C': np.arange(1, 10,1)
                 }
    
    # Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=SVM_model, 
                                  param_distributions = parameters,
                                  scoring = 'neg_mean_squared_error',
                                  cv = 3,n_iter=100,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    SVM_model_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    SVM_model_best.fit(X_train, Y_train)
    SVM_model_score = cross_val_score(estimator = SVM_model_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    SVM_model_score = (np.sqrt(np.abs(SVM_model_score)))
    
    SVM_model_score_mean = SVM_model_score.mean()
    SVM_model_score_std  = SVM_model_score.std()
  

  #.6.KNN
    KNN_model=KNeighborsRegressor() 
    KNN_model.fit(X_train, Y_train)
    KNN_model_score = cross_val_score(estimator = KNN_model, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    KNN_model_score = (np.sqrt(np.abs(KNN_model_score)))
    
    KNN_model_score_mean = KNN_model_score.mean()
    KNN_model_score_std  = KNN_model_score.std()



    # Choose some parameter combinations to try
    parameters = { 'n_neighbors': np.arange(1, 31, 1),
	              'metric': ["minkowski"]
                 }
    
   #Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=KNN_model, 
                                  param_distributions = parameters,
                                  scoring = 'neg_mean_squared_error',
                                  cv = 10,n_iter=30,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    KNN_model_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    KNN_model_best.fit(X_train, Y_train)
    KNN_model_score = cross_val_score(estimator = KNN_model_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    KNN_model_score = (np.sqrt(np.abs(KNN_model_score)))
    
    KNN_model_score_mean = KNN_model_score.mean()
    KNN_model_score_std  = KNN_model_score.std()
    
    
    
 
    

 #---------------Step 9:Prediction on test data -------------------------------------------------------
     Y_pred1 = logreg.predict(X_test)
     Y_pred2 = decision_tree_best.predict(X_test)
     Y_pred3 = random_forest_best.predict(X_test)
     Y_pred4 = Xgboost_best.predict(X_test)
     Y_pred5 = SVM_Classifier_best.predict(X_test)
     Y_pred6 = KNN_Classifier_best.predict(X_test)
    
    
    
     
    submission = pd.DataFrame({
            "User_ID": test_df["User_ID"],
            "Product_ID": test_df["Product_ID"],
            "Purchase": Y_pred4
        })
    submission=submission[["User_ID","Product_ID","Purchase"]]
    
    submission.to_csv('submission.csv', index=False)





 
 