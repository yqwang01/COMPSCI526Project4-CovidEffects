import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")
import sys
sys.stdout.reconfigure(encoding='utf-8')


excel_path = './Coursework/COMPSCI526_Data_scicence/Project/Dataset/COVID-19-Constructed-Dataset-(PANEL).xlsx'
df = pd.read_excel(excel_path)
txt_path = './Coursework/COMPSCI526_Data_scicence/Project/Results/casual_inference_yiqing.txt'
txt_file = open(txt_path, 'w', encoding='utf-8', errors='ignore')


factors = ['school', 'gradelevel', 'gender', 'covidpos', 'householdincome', 'freelunch', 'numcomputers', 'familysize', 'fathereduc', 'mothereduc', 'timeperiod']
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


outstring = ">>>>>>>>>Original Data\n"
txt_file.write(outstring)
print(outstring)
for factor in factors:
    outstring = ">>>>>>>>>Factor: " + factor + "\n"
    txt_file.write(outstring)
    print(outstring)
    other_factors = [x for x in factors if x != factor]
    
    outstring = "See other factors as common causes.\n"
    txt_file.write(outstring)
    print(outstring)
    model = CausalModel(
        data=df,
        treatment=factor,
        outcome='meanscore',
        common_causes=other_factors
    )
    identified_estimand = model.identify_effect()
    print(identified_estimand)
    txt_file.write(str(identified_estimand) + "\n")
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    outstring = f"Estimated Causal Effect: {estimate.value}"
    txt_file.write(outstring + "\n")
    print(outstring)
    res = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    print(res)
    txt_file.write(str(res) + "\n")
    
    # outstring = "See other factors as effect modifiers.\n"
    # print(outstring)
    # txt_file.write(outstring)
    # model = CausalModel(
    #     data=df,
    #     treatment=factor,
    #     outcome='meanscore',
    #     effect_modifiers=other_factors,
    # )
    # identified_estimand = model.identify_effect()
    # print(identified_estimand)
    # txt_file.write(str(identified_estimand) + "\n")
    # estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    # outstring = f"Estimated Causal Effect: {estimate.value}"
    # txt_file.write(outstring + "\n")
    # print(outstring)
    # res = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    # print(res)
    # txt_file.write(str(res) + "\n")
    
    
outstring = ">>>>>>>>>Average Data\n"
txt_file.write(outstring)
print(outstring)
for factor in non_time_factors:
    outstring = ">>>>>>>>>Factor: " + factor + "\n"
    txt_file.write(outstring)
    print(outstring)
    other_factors = [x for x in non_time_factors if x != factor]
    
    outstring = "See other factors as common causes.\n"
    txt_file.write(outstring)
    print(outstring)
    model = CausalModel(
        data=df_average,
        treatment=factor,
        outcome='meanscore',
        common_causes=other_factors
    )
    identified_estimand = model.identify_effect()
    print(identified_estimand)
    txt_file.write(str(identified_estimand) + "\n")
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    outstring = f"Estimated Causal Effect: {estimate.value}"
    txt_file.write(outstring + "\n")
    print(outstring)
    res = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    print(res)
    txt_file.write(str(res) + "\n")
    
    # outstring = "See other factors as effect modifiers.\n"
    # txt_file.write(outstring)
    # print(outstring)
    # model = CausalModel(
    #     data=df_average,
    #     treatment=factor,
    #     outcome='meanscore',
    #     effect_modifiers=other_factors,
    # )
    # identified_estimand = model.identify_effect()
    # print(identified_estimand)
    # txt_file.write(str(identified_estimand) + "\n")
    # estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    # outstring = f"Estimated Causal Effect: {estimate.value}"
    # txt_file.write(outstring + "\n")
    # print(outstring)
    # res = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    # print(res)
    # txt_file.write(str(res) + "\n")

outstring = ">>>>>>>>>Pre-COVID Data\n"
txt_file.write(outstring)
print(outstring)
for factor in non_time_factors:
    outstring = ">>>>>>>>>Factor: " + factor + "\n"
    txt_file.write(outstring)
    print(outstring)
    other_factors = [x for x in non_time_factors if x != factor]
    
    outstring = "See other factors as common causes.\n"
    txt_file.write(outstring)
    print(outstring)
    model = CausalModel(
        data=df_precovid,
        treatment=factor,
        outcome='meanscore',
        common_causes=other_factors
    )
    identified_estimand = model.identify_effect()
    print(identified_estimand)
    txt_file.write(str(identified_estimand) + "\n")
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    outstring = f"Estimated Causal Effect: {estimate.value}"
    txt_file.write(outstring + "\n")
    print(outstring)
    res = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    print(res)
    txt_file.write(str(res) + "\n")
    
    # outstring = "See other factors as effect modifiers.\n"
    # txt_file.write(outstring)
    # print(outstring)
    # model = CausalModel(
    #     data=df_precovid,
    #     treatment=factor,
    #     outcome='meanscore',
    #     effect_modifiers=other_factors,
    # )
    # identified_estimand = model.identify_effect()
    # print(identified_estimand)
    # txt_file.write(str(identified_estimand) + "\n")
    # estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    # outstring = f"Estimated Causal Effect: {estimate.value}"
    # txt_file.write(outstring + "\n")
    # print(outstring)
    # res = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    # print(res)
    # txt_file.write(str(res) + "\n")


outstring = ">>>>>>>>>Post-COVID Data\n"
txt_file.write(outstring)
print(outstring)
for factor in non_time_factors:
    outstring = ">>>>>>>>>Factor: " + factor + "\n"
    txt_file.write(outstring)
    print(outstring)
    other_factors = [x for x in non_time_factors if x != factor]
    
    outstring = "See other factors as common causes.\n"
    txt_file.write(outstring)
    print(outstring)
    model = CausalModel(
        data=df_postcovid,
        treatment=factor,
        outcome='meanscore',
        common_causes=other_factors
    )
    identified_estimand = model.identify_effect()
    print(identified_estimand)
    txt_file.write(str(identified_estimand) + "\n")
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    outstring = f"Estimated Causal Effect: {estimate.value}"
    txt_file.write(outstring + "\n")
    print(outstring)
    res = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    print(res)
    txt_file.write(str(res) + "\n")
    
    # outstring = "See other factors as effect modifiers.\n"
    # txt_file.write(outstring)
    # print(outstring)
    # model = CausalModel(
    #     data=df_postcovid,
    #     treatment=factor,
    #     outcome='meanscore',
    #     effect_modifiers=other_factors,
    # )
    # identified_estimand = model.identify_effect()
    # print(identified_estimand)
    # txt_file.write(str(identified_estimand) + "\n")
    # estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    # outstring = f"Estimated Causal Effect: {estimate.value}"
    # txt_file.write(outstring + "\n")
    # print(outstring)
    # res = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    # print(res)
    # txt_file.write(str(res) + "\n")


txt_file.close()
