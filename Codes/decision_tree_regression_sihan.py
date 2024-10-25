# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:36:35 2024

@author: 13190
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score

def calculate_metrics(confusion_matrix):
    TP = confusion_matrix[0, 0]  # True Positive
    FP = confusion_matrix[0, 1]  # False Positive
    FN = confusion_matrix[1, 0]  # False Negative
    TN = confusion_matrix[1, 1]  # True Negative

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    return accuracy, precision, recall, f1_score



dataset_path = 'C:/lsh/learn_sth/data_science_ece/project/data/Covid_dataset.csv'
df = pd.read_csv(dataset_path)

all_feature_list = ['school','gradelevel','gender','covidpos','householdincome','freelunch','numcomputers','familysize','fathereduc','mothereduc']
data_dict = {}
for feature in all_feature_list:
    data_dict[feature] = df[df['timeperiod']==1][feature].tolist() # 0 or 1

all_scores = ['readingscore','writingscore','mathscore','readingscoreSL','writingscoreSL','mathscoreSL']
average_score,average_score_InPerson,average_score_Online = [],[],[],

for student_id in range(1,1401):
    
    temp_InPerson = df[(df['studentID']==student_id) & 
              ((df['timeperiod']==0) | (df['timeperiod']==1) | (df['timeperiod']==2))]
    InPerson_mean = temp_InPerson[all_scores].mean(axis=1).mean()
    average_score_InPerson.append(InPerson_mean)
    
    temp_Online = df[(df['studentID']==student_id) & 
              ((df['timeperiod']==3) | (df['timeperiod']==4) | (df['timeperiod']==5))]
    Online_mean = temp_Online[all_scores].mean(axis=1).mean()
    average_score_Online.append(Online_mean)
    
    average_score.append((InPerson_mean+Online_mean)/2)


data_dict['average_score'] = average_score
data_dict['average_score_InPerson'] = average_score_InPerson
data_dict['average_score_Online'] = average_score_Online
    
    
data = pd.DataFrame(data_dict)

# Features and target
outputs = ['average_score','average_score_InPerson','average_score_Online']

X = data.drop(columns=outputs)
p = 1

outputs = ['average_score_InPerson','average_score_Online']
#%% decision tree regression
for output in outputs:
    if output == 'average_score_InPerson':
        before_after = 'before'
    else:
        before_after = 'after'
    
    y = data[output]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree = DecisionTreeRegressor(random_state=42)
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas  
    
    tree_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid={'ccp_alpha': ccp_alphas}, cv=5)
    tree_cv.fit(X_train, y_train)
    best_alpha = tree_cv.best_params_['ccp_alpha']
    # print('the best alpha selected by cross validation is: {}'.format(best_alpha))
    
    
    prunned_tree = DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
    prunned_tree.fit(X_train, y_train)
    
    
    y_pred = prunned_tree.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    rss = mse * len(y_test)
    rse = np.sqrt(rss / (1400 - p - 1))

    print("Regression on {}. Training MSE: {} , R2: {}, RSS: {}, RSE: {}".format(output,mse,r2, rss, rse))
    
    
    # Make predictions on the test data
    y_pred = prunned_tree.predict(X_test)
    
    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rss = mse * len(y_test)
    rse = np.sqrt(rss / (1400 - p - 1))

    print("Regression on {}. Testing MSE: {} , R2: {}, RSS: {}, RSE: {}".format(output,mse,r2, rss, rse))
    

    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
    # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal Fit')  # Diagonal line
    plt.plot([30,100], [30,100], color='red', lw=2, label='Ideal Fit')  # Diagonal line
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.xlim([30,100])
    plt.ylim([30,100])
    plt.title('Decision Tree Regressor: Actual vs Predicted Values (average score {} lockdown)'.format(before_after))
    plt.legend()
    plt.grid(True)
    plt.savefig('DecisionTree_{}Lockdown.jpg'.format(before_after))
    plt.clf()

#%% random forest regression

for output in outputs:
    
    if output == 'average_score_InPerson':
        before_after = 'before'
    else:
        before_after = 'after'
        
    y = data[output]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree = RandomForestRegressor(random_state=42)
    
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, 30], #[None, 10, 20, 30]
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    # 'bootstrap': [True, False]
    }
    
    grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)
    
    
    y_pred = best_model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    rss = mse * len(y_test)
    rse = np.sqrt(rss / (1400 - p - 1))

    print("Regression on {}. Training MSE: {} , R2: {}, RSS: {}, RSE: {}".format(output,mse,r2, rss, rse))
    
    
    # Make predictions on the test data
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rss = mse * len(y_test)
    rse = np.sqrt(rss / (1400 - p - 1))

    print("Regression on {}. Testing MSE: {} , R2: {}, RSS: {}, RSE: {}".format(output,mse,r2, rss, rse))
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
    # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal Fit')  # Diagonal line
    plt.plot([30,100], [30,100], color='red', lw=2, label='Ideal Fit')  # Diagonal line
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('Random Forest Regressor: Actual vs Predicted Values (average score {} lockdown)'.format(before_after))
    plt.legend()
    plt.xlim([30,100])
    plt.ylim([30,100])
    plt.grid(True)
    plt.savefig('RandomForest_{}Lockdown.jpg'.format(before_after))
    plt.clf()

#%%




kf = KFold(n_splits=5, shuffle=True, random_state=42)
cm_all = {}

for output in outputs:
    y = np.array(data[output]>np.mean(data[output])).astype(int)
    cm_all[output] = [[0,0],[0,0]]
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Use .iloc for DataFrame
        y_train, y_test = y[train_index], y[test_index]
        
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        param_grid = {
            'max_depth': [3, 5, 7, None],  # Maximum depth of the tree
            'min_samples_split': [2, 10, 20],  # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 5, 10],  # Minimum number of samples required at a leaf node
            'criterion': ['gini', 'entropy']  # The function to measure the quality of a split
        }
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
    
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        # y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_all[output] += cm
        
    print('Confusion matrix for {}'.format(output))
    print(cm_all[output])
    accuracy, precision, recall, f1 = calculate_metrics(cm_all[output])
    print('accuracy:{}, precision:{}, reacall:{}, f1:{}'.format(accuracy, precision, recall, f1))
    print("\n")
#%%


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cm_all = {}

for output in outputs:
    y = np.array(data[output]>np.mean(data[output])).astype(int)
    cm_all[output] = [[0,0],[0,0]]
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Use .iloc for DataFrame
        y_train, y_test = y[train_index], y[test_index]
        
        
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        param_grid = {
            'n_estimators': [10, 50, 100],  # Number of trees in the forest
            'max_depth': [3, 5, 7, None],  # Maximum depth of the tree
            'min_samples_split': [2, 10, 20],  # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 5, 10],  # Minimum number of samples required at a leaf node
            'criterion': ['gini', 'entropy']  # The function to measure the quality of a split
        }
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=1)
    
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_all[output] += cm
        

    print('Confusion matrix for {}'.format(output))
    print(cm_all[output])
    accuracy, precision, recall, f1 = calculate_metrics(cm_all[output])
    print('accuracy:{}, precision:{}, reacall:{}, f1:{}'.format(accuracy, precision, recall, f1))
    print("\n")

#%%


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cm_all = {}

for output in outputs:
    y = np.array(data[output]>np.mean(data[output])).astype(int)
    cm_all[output] = [[0,0],[0,0]]
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        # Split the data: 4 parts for training, 1 part for testing
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Use .iloc for DataFrame
        y_train, y_test = y[train_index], y[test_index]
        
    
        clf = SVC(random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization parameter
            'kernel': ['rbf', 'poly', 'sigmoid'],  # Type of kernel 'linear' is too slow
            # 'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
            # 'class_weight': [None, 'balanced']  # Class weight adjustment
        }
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=1)
    
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        # y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_all[output] += cm
        
    print('Confusion matrix for {}'.format(output))
    print(cm_all[output])
    accuracy, precision, recall, f1 = calculate_metrics(cm_all[output])
    print('accuracy:{}, precision:{}, reacall:{}, f1:{}'.format(accuracy, precision, recall, f1))
    print("\n")