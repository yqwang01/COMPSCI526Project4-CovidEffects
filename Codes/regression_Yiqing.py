import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, f1_score, classification_report, mean_squared_error, confusion_matrix


excel_path = './Coursework/COMPSCI526_Data_scicence/Project/Dataset/COVID-19-Constructed-Dataset-(PANEL).xlsx'
df = pd.read_excel(excel_path)
txt_path = './Coursework/COMPSCI526_Data_scicence/Project/Results/regression_yiqing.txt'
txt_file = open(txt_path, 'w')


factors = ['school', 'gradelevel', 'gender', 'covidpos', 'householdincome', 'freelunch', 'numcomputers', 'familysize', 'fathereduc', 'mothereduc', 'timeperiod']
key_factors = ['school', 'gender', 'covidpos', 'householdincome', 'freelunch', 'numcomputers', 'fathereduc', 'mothereduc', 'timeperiod']
non_time_factors = ['school', 'gradelevel', 'gender', 'covidpos', 'householdincome', 'freelunch', 'numcomputers', 'familysize', 'fathereduc', 'mothereduc']
scores = ['readingscore', 'writingscore', 'mathscore', 'readingscoreSL', 'writingscoreSL', 'mathscoreSL', 'meanscore']
for factor in factors:
    data_factor = df[factor]
    if factor == 'householdincome':
        data_factor = data_factor / 180000
    df[factor] = data_factor
df['meanscore'] = df[scores[:-1]].mean(axis=1)


df_average = df.groupby('studentID').agg(
    school=pd.NamedAgg(column='school', aggfunc='first'),
    gradelevel=pd.NamedAgg(column='gradelevel', aggfunc='first'),
    gender=pd.NamedAgg(column='gender', aggfunc='first'),
    covidpos=pd.NamedAgg(column='covidpos', aggfunc='first'),
    householdincome=pd.NamedAgg(column='householdincome', aggfunc='first'),
    freelunch=pd.NamedAgg(column='freelunch', aggfunc='first'),
    numcomputers=pd.NamedAgg(column='numcomputers', aggfunc='first'),
    familysize=pd.NamedAgg(column='familysize', aggfunc='first'),
    fathereduc=pd.NamedAgg(column='fathereduc', aggfunc='first'),
    mothereduc=pd.NamedAgg(column='mothereduc', aggfunc='first'),
    readingscore=pd.NamedAgg(column='readingscore', aggfunc='mean'),
    writingscore=pd.NamedAgg(column='writingscore', aggfunc='mean'),
    mathscore=pd.NamedAgg(column='mathscore', aggfunc='mean'),
    readingscoreSL=pd.NamedAgg(column='readingscoreSL', aggfunc='mean'),
    writingscoreSL=pd.NamedAgg(column='writingscoreSL', aggfunc='mean'),
    mathscoreSL=pd.NamedAgg(column='mathscoreSL', aggfunc='mean'),
    meanscore=pd.NamedAgg(column='meanscore', aggfunc='mean'),
).reset_index()
df_precovid = df[df['timeperiod'] <= 2].groupby('studentID').agg(
    school=pd.NamedAgg(column='school', aggfunc='first'),
    gradelevel=pd.NamedAgg(column='gradelevel', aggfunc='first'),
    gender=pd.NamedAgg(column='gender', aggfunc='first'),
    covidpos=pd.NamedAgg(column='covidpos', aggfunc='first'),
    householdincome=pd.NamedAgg(column='householdincome', aggfunc='first'),
    freelunch=pd.NamedAgg(column='freelunch', aggfunc='first'),
    numcomputers=pd.NamedAgg(column='numcomputers', aggfunc='first'),
    familysize=pd.NamedAgg(column='familysize', aggfunc='first'),
    fathereduc=pd.NamedAgg(column='fathereduc', aggfunc='first'),
    mothereduc=pd.NamedAgg(column='mothereduc', aggfunc='first'),
    readingscore=pd.NamedAgg(column='readingscore', aggfunc='mean'),
    writingscore=pd.NamedAgg(column='writingscore', aggfunc='mean'),
    mathscore=pd.NamedAgg(column='mathscore', aggfunc='mean'),
    readingscoreSL=pd.NamedAgg(column='readingscoreSL', aggfunc='mean'),
    writingscoreSL=pd.NamedAgg(column='writingscoreSL', aggfunc='mean'),
    mathscoreSL=pd.NamedAgg(column='mathscoreSL', aggfunc='mean'),
    meanscore=pd.NamedAgg(column='meanscore', aggfunc='mean'),
).reset_index()
df_postcovid = df[df['timeperiod'] > 2].groupby('studentID').agg(
    school=pd.NamedAgg(column='school', aggfunc='first'),
    gradelevel=pd.NamedAgg(column='gradelevel', aggfunc='first'),
    gender=pd.NamedAgg(column='gender', aggfunc='first'),
    covidpos=pd.NamedAgg(column='covidpos', aggfunc='first'),
    householdincome=pd.NamedAgg(column='householdincome', aggfunc='first'),
    freelunch=pd.NamedAgg(column='freelunch', aggfunc='first'),
    numcomputers=pd.NamedAgg(column='numcomputers', aggfunc='first'),
    familysize=pd.NamedAgg(column='familysize', aggfunc='first'),
    fathereduc=pd.NamedAgg(column='fathereduc', aggfunc='first'),
    mothereduc=pd.NamedAgg(column='mothereduc', aggfunc='first'),
    readingscore=pd.NamedAgg(column='readingscore', aggfunc='mean'),
    writingscore=pd.NamedAgg(column='writingscore', aggfunc='mean'),
    mathscore=pd.NamedAgg(column='mathscore', aggfunc='mean'),
    readingscoreSL=pd.NamedAgg(column='readingscoreSL', aggfunc='mean'),
    writingscoreSL=pd.NamedAgg(column='writingscoreSL', aggfunc='mean'),
    mathscoreSL=pd.NamedAgg(column='mathscoreSL', aggfunc='mean'),
    meanscore=pd.NamedAgg(column='meanscore', aggfunc='mean'),
).reset_index()


outstring = ">>>>>>Original Scores>>>>>>\n"
print(outstring)
txt_file.write(outstring)
factor_data = df[factors]
# key_data = df[key_factors]
for score in scores:
    outstring = f">>>>>>Score: {score}>>>>>>\n"
    print(outstring)
    txt_file.write(outstring)
    
    outstring = ">>>>>>Linear Regression with All Factors\n"
    print(outstring)
    txt_file.write(outstring)
    model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(factor_data, df[score], test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    outstring = f"Coefficients: {model.coef_}, Intercept: {model.intercept_}\n"
    print(outstring)
    txt_file.write(outstring)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rss_train = np.sum((pred_train - y_train)**2)
    rss_test = np.sum((pred_test - y_test)**2)
    rse_train = np.sqrt(rss_train) / (len(pred_train) - len(model.coef_) - 1)
    rse_test = np.sqrt(rss_test) / (len(pred_test) - len(model.coef_) - 1)
    outstring = f"MSE_train: {mse_train}, MSE_test: {mse_test}, R2_train: {r2_train}, R2_test: {r2_test}, RSS_train: {rss_train}, RSS_test: {rss_test}, RSE_train: {rse_train}, RSE_test: {rse_test}\n"
    print(outstring)
    txt_file.write(outstring)
    
    # outstring = ">>>>>>Linear Regression with Key Factors\n"
    # print(outstring)
    # txt_file.write(outstring)
    # model = LinearRegression()
    # x_train, x_test, y_train, y_test = train_test_split(key_data, df[score], test_size=0.2, random_state=0)
    # model.fit(x_train, y_train)
    # outstring = f"Coefficients: {model.coef_}, Intercept: {model.intercept_}\n"
    # print(outstring)
    # txt_file.write(outstring)
    # pred_train = model.predict(x_train)
    # pred_test = model.predict(x_test)
    # r2_train = r2_score(y_train, pred_train)
    # r2_test = r2_score(y_test, pred_test)
    # rss_train = np.sum((pred_train - y_train)**2)
    # rss_test = np.sum((pred_test - y_test)**2)
    # rse_train = np.sqrt(rss_train) / (len(pred_train) - len(model.coef_) - 1)
    # rse_test = np.sqrt(rss_test) / (len(pred_test) - len(model.coef_) - 1)
    # outstring = f"R2_train: {r2_train}, R2_test: {r2_test}, RSS_train: {rss_train}, RSS_test: {rss_test}, RSE_train: {rse_train}, RSE_test: {rse_test}\n"
    # print(outstring)
    # txt_file.write(outstring)
    
    outstring = ">>>>>>Logistic Regression with All Factors\n"
    print(outstring)
    txt_file.write(outstring)
    score_median = np.median(df[score])
    outstring = f"Score Median: {score_median}\n"
    print(outstring)
    txt_file.write(outstring)
    y_binary = (df[score] > score_median).astype(int)
    model = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(factor_data, y_binary, test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    train_accuracy = np.mean(pred_train == y_train)
    test_accuracy = np.mean(pred_test == y_test)
    train_f1 = f1_score(y_train, pred_train)
    test_f1 = f1_score(y_test, pred_test)
    conf_matrix = confusion_matrix(y_test, pred_test)
    report = classification_report(y_test, pred_test)
    outstring = f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"Train F1 Score: {train_f1}, Test F1 Score: {test_f1}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = "Confusion Matrix:\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"{conf_matrix}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"{report}\n"
    print(outstring)
    txt_file.write(outstring)
    
    # outstring = ">>>>>>Logistic Regression with Key Factors\n"
    # print(outstring)
    # txt_file.write(outstring)
    # score_median = np.median(df[score])
    # outstring = f"Score Median: {score_median}\n"
    # print(outstring)
    # txt_file.write(outstring)
    # y_binary = (df[score] > score_median).astype(int)
    # model = LogisticRegression()
    # x_train, x_test, y_train, y_test = train_test_split(key_data, y_binary, test_size=0.2, random_state=0)
    # model.fit(x_train, y_train)
    # pred_train = model.predict(x_train)
    # pred_test = model.predict(x_test)
    # train_accuracy = np.mean(pred_train == y_train)
    # test_accuracy = np.mean(pred_test == y_test)
    # train_f1 = f1_score(y_train, pred_train)
    # test_f1 = f1_score(y_test, pred_test)
    # report = classification_report(y_test, pred_test)
    # outstring = f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n"
    # print(outstring)
    # txt_file.write(outstring)
    # outstring = f"Train F1 Score: {train_f1}, Test F1 Score: {test_f1}\n"
    # print(outstring)
    # txt_file.write(outstring)
    # outstring = f"{report}\n"
    # print(outstring)
    # txt_file.write(outstring)


outstring = ">>>>>>Average Scores>>>>>>\n"
print(outstring)
txt_file.write(outstring)
factor_data = df_average[non_time_factors]
for score in scores:
    outstring = f">>>>>>Score: {score}>>>>>>\n"
    print(outstring)
    txt_file.write(outstring)
    
    outstring = ">>>>>>Linear Regression\n"
    print(outstring)
    txt_file.write(outstring)
    model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(factor_data, df_average[score], test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    outstring = f"Coefficients: {model.coef_}, Intercept: {model.intercept_}\n"
    print(outstring)
    txt_file.write(outstring)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rss_train = np.sum((pred_train - y_train)**2)
    rss_test = np.sum((pred_test - y_test)**2)
    rse_train = np.sqrt(rss_train) / (len(pred_train) - len(model.coef_) - 1)
    rse_test = np.sqrt(rss_test) / (len(pred_test) - len(model.coef_) - 1)
    outstring = f"MSE_train: {mse_train}, MSE_test: {mse_test}, R2_train: {r2_train}, R2_test: {r2_test}, RSS_train: {rss_train}, RSS_test: {rss_test}, RSE_train: {rse_train}, RSE_test: {rse_test}\n"
    print(outstring)
    txt_file.write(outstring)
    
    outstring = ">>>>>>Logistic Regression\n"
    print(outstring)
    txt_file.write(outstring)
    score_median = np.median(df_average[score])
    outstring = f"Score Median: {score_median}\n"
    print(outstring)
    txt_file.write(outstring)
    y_binary = (df_average[score] > score_median).astype(int)
    model = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(factor_data, y_binary, test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    train_accuracy = np.mean(pred_train == y_train)
    test_accuracy = np.mean(pred_test == y_test)
    train_f1 = f1_score(y_train, pred_train)
    test_f1 = f1_score(y_test, pred_test)
    conf_matrix = confusion_matrix(y_test, pred_test)
    report = classification_report(y_test, pred_test)
    outstring = f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"Train F1 Score: {train_f1}, Test F1 Score: {test_f1}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = "Confusion Matrix:\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"{conf_matrix}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"{report}\n"
    print(outstring)
    txt_file.write(outstring)


outstring = ">>>>>>Pre-COVID Scores>>>>>>\n"
print(outstring)
txt_file.write(outstring)
factor_data = df_precovid[non_time_factors]
for score in scores:
    outstring = f">>>>>>Score: {score}>>>>>>\n"
    print(outstring)
    txt_file.write(outstring)
    
    outstring = ">>>>>>Linear Regression\n"
    print(outstring)
    txt_file.write(outstring)
    model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(factor_data, df_precovid[score], test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    outstring = f"Coefficients: {model.coef_}, Intercept: {model.intercept_}\n"
    print(outstring)
    txt_file.write(outstring)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rss_train = np.sum((pred_train - y_train)**2)
    rss_test = np.sum((pred_test - y_test)**2)
    rse_train = np.sqrt(rss_train) / (len(pred_train) - len(model.coef_) - 1)
    rse_test = np.sqrt(rss_test) / (len(pred_test) - len(model.coef_) - 1)
    outstring = f"MSE_train: {mse_train}, MSE_test: {mse_test}, R2_train: {r2_train}, R2_test: {r2_test}, RSS_train: {rss_train}, RSS_test: {rss_test}, RSE_train: {rse_train}, RSE_test: {rse_test}\n"
    print(outstring)
    txt_file.write(outstring)
    
    outstring = ">>>>>>Logistic Regression\n"
    print(outstring)
    txt_file.write(outstring)
    score_median = np.median(df_precovid[score])
    outstring = f"Score Median: {score_median}\n"
    print(outstring)
    txt_file.write(outstring)
    y_binary = (df_precovid[score] > score_median).astype(int)
    model = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(factor_data, y_binary, test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    train_accuracy = np.mean(pred_train == y_train)
    test_accuracy = np.mean(pred_test == y_test)
    train_f1 = f1_score(y_train, pred_train)
    test_f1 = f1_score(y_test, pred_test)
    conf_matrix = confusion_matrix(y_test, pred_test)
    report = classification_report(y_test, pred_test)
    outstring = f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"Train F1 Score: {train_f1}, Test F1 Score: {test_f1}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = "Confusion Matrix:\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"{conf_matrix}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"{report}\n"
    print(outstring)
    txt_file.write(outstring)


outstring = ">>>>>>Post-COVID Scores>>>>>>\n"
print(outstring)
txt_file.write(outstring)
factor_data = df_postcovid[non_time_factors]
for score in scores:
    outstring = f">>>>>>Score: {score}>>>>>>\n"
    print(outstring)
    txt_file.write(outstring)
    
    outstring = ">>>>>>Linear Regression\n"
    print(outstring)
    txt_file.write(outstring)
    model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(factor_data, df_postcovid[score], test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    outstring = f"Coefficients: {model.coef_}, Intercept: {model.intercept_}\n"
    print(outstring)
    txt_file.write(outstring)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rss_train = np.sum((pred_train - y_train)**2)
    rss_test = np.sum((pred_test - y_test)**2)
    rse_train = np.sqrt(rss_train) / (len(pred_train) - len(model.coef_) - 1)
    rse_test = np.sqrt(rss_test) / (len(pred_test) - len(model.coef_) - 1)
    outstring = f"MSE_train: {mse_train}, MSE_test: {mse_test}, R2_train: {r2_train}, R2_test: {r2_test}, RSS_train: {rss_train}, RSS_test: {rss_test}, RSE_train: {rse_train}, RSE_test: {rse_test}\n"
    print(outstring)
    txt_file.write(outstring)
    
    outstring = ">>>>>>Logistic Regression\n"
    print(outstring)
    txt_file.write(outstring)
    score_median = np.median(df_postcovid[score])
    outstring = f"Score Median: {score_median}\n"
    print(outstring)
    txt_file.write(outstring)
    y_binary = (df_postcovid[score] > score_median).astype(int)
    model = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(factor_data, y_binary, test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    train_accuracy = np.mean(pred_train == y_train)
    test_accuracy = np.mean(pred_test == y_test)
    train_f1 = f1_score(y_train, pred_train)
    test_f1 = f1_score(y_test, pred_test)
    conf_matrix = confusion_matrix(y_test, pred_test)
    report = classification_report(y_test, pred_test)
    outstring = f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"Train F1 Score: {train_f1}, Test F1 Score: {test_f1}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = "Confusion Matrix:\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"{conf_matrix}\n"
    print(outstring)
    txt_file.write(outstring)
    outstring = f"{report}\n"
    print(outstring)
    txt_file.write(outstring)



txt_file.close()