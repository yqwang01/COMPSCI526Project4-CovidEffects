import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


excel_path = './Coursework/COMPSCI526_Data_scicence/Project/Dataset/COVID-19-Constructed-Dataset-(PANEL).xlsx'
df = pd.read_excel(excel_path)
txt_path = './Coursework/COMPSCI526_Data_scicence/Project/Codes/basic_analysis_yiqing.txt'
txt_file = open(txt_path, 'w')
save_dir = './Coursework/COMPSCI526_Data_scicence/Project/Results/FactorAnalysis/Visualizations_Yiqing/'
factors = ['numcomputers', 'familysize']
scores = ['readingscore', 'writingscore', 'mathscore', 'readingscoreSL', 'writingscoreSL', 'mathscoreSL']



for factor in factors:
    for score in scores:
        
        data_score_factor = df.groupby(factor)[score]
        mean_var_table = data_score_factor.agg(['mean', 'var'])
        outstring = '\n' + "-" * 5 + 'Factor: ' + factor + ', Score:' + score + '-' * 5 + "\n"
        print(outstring)
        txt_file.write(outstring)
        outstring = "Mean and Variance of {} by {}\n".format(score, factor)
        print(outstring)
        txt_file.write(outstring)
        print(mean_var_table)
        txt_file.write(mean_var_table.to_string())
        
        outstring = "\n>>>ANOVA test for {} by {}\n".format(score, factor)
        print(outstring)
        txt_file.write(outstring)
        group_data = [group for name, group in data_score_factor]
        f_value, p_value = stats.f_oneway(*group_data)
        outstring = "F value: {}, p value: {}\n".format(f_value, p_value)
        print(outstring)
        txt_file.write(outstring)
        if p_value < 0.05:
            outstring = "The mean {} scores of different groups of {} are significantly different, the null hypothesis is rejected.\n".format(score, factor)
            print(outstring)
            txt_file.write(outstring)
        else:
            outstring = "The mean {} scores of different groups of {} are not significantly different, the null hypothesis is not rejected.\n".format(score, factor)
            print(outstring)
            txt_file.write(outstring)
        
        outstring = ">>>Spearman correlation between {} and {}\n".format(factor, score)
        print(outstring)
        txt_file.write(outstring)
        spearman_corr, p_value = stats.spearmanr(df[factor], df[score])
        outstring = "Spearman correlation: {}, p value: {}\n".format(spearman_corr, p_value)
        print(outstring)
        txt_file.write(outstring)
        if p_value < 0.05:
            outstring = "The correlation between {} and {} is significant, the null hypothesis is rejected.\n".format(factor, score)
            print(outstring)
            txt_file.write(outstring)
        else:
            outstring = "The correlation between {} and {} is not significant, the null hypothesis is not rejected.\n".format(factor, score)
            print(outstring)
            txt_file.write(outstring)
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=factor, y=score)
        plt.title('Boxplot of {} by {}'.format(score, factor))
        plt.xlabel(factor)
        plt.ylabel(score)
        plt.savefig(save_dir + '{}_by_{}.png'.format(score, factor))
        plt.close()
                
            

outstring = '\n' + "-" * 10 + "\n"
print(outstring)
txt_file.write(outstring)
outstring = "Pre-COVID-19\n"
print(outstring)
txt_file.write(outstring)
outstring = "-" * 10 + "\n"
df_precovid = df[df['timeperiod'] <= 2]
for factor in factors:
    for score in scores:
        data_score_factor = df_precovid.groupby(factor)[score]
        mean_var_table = data_score_factor.agg(['mean', 'var'])
        outstring = '\n' + "-" * 5 + 'Factor: ' + factor + ', Score:' + score + '-' * 5 + "\n"
        print(outstring)
        txt_file.write(outstring)
        outstring = "Mean and Variance of {} by {}\n".format(score, factor)
        print(outstring)
        txt_file.write(outstring)
        print(mean_var_table)
        txt_file.write(mean_var_table.to_string())
        
        outstring = "\n>>>ANOVA test for {} by {}\n".format(score, factor)
        group_data = [group for name, group in data_score_factor]
        f_value, p_value = stats.f_oneway(*group_data)
        outstring = "F value: {}, p value: {}\n".format(f_value, p_value)
        print(outstring)
        txt_file.write(outstring)
        if p_value < 0.05:
            outstring = "The mean {} scores of different groups of {} are significantly different, the null hypothesis is rejected.\n".format(score, factor)
            print(outstring)
            txt_file.write(outstring)
        else:
            outstring = "The mean {} scores of different groups of {} are not significantly different, the null hypothesis is not rejected.\n".format(score, factor)
            print(outstring)
            txt_file.write(outstring)
        
        outstring = ">>>Spearman correlation between {} and {}\n".format(factor, score)
        print(outstring)
        txt_file.write(outstring)
        spearman_corr, p_value = stats.spearmanr(df_precovid[factor], df_precovid[score])
        outstring = "Spearman correlation: {}, p value: {}\n".format(spearman_corr, p_value)
        print(outstring)
        txt_file.write(outstring)
        if p_value < 0.05:
            outstring = "The correlation between {} and {} is significant, the null hypothesis is rejected.\n".format(factor, score)
            print(outstring)
            txt_file.write(outstring)
        else:
            outstring = "The correlation between {} and {} is not significant, the null hypothesis is not rejected.\n".format(factor, score)
            print(outstring)
            txt_file.write(outstring)
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_precovid, x=factor, y=score)
        plt.title('Boxplot of {} by {}, pre-covid'.format(score, factor))
        plt.xlabel(factor)
        plt.ylabel(score)
        plt.savefig(save_dir + 'precovid_{}_by_{}.png'.format(score, factor))
        plt.close()


outstring = '\n' + "-" * 10 + "\n"
print(outstring)
txt_file.write(outstring)
outstring = "Post-COVID-19\n"
print(outstring)
txt_file.write(outstring)
outstring = "-" * 10 + "\n"
df_postcovid = df[df['timeperiod'] > 2]

for factor in factors:
    for score in scores:
        data_score_factor = df_postcovid.groupby(factor)[score]
        mean_var_table = data_score_factor.agg(['mean', 'var'])
        outstring = '\n' + "-" * 5 + 'Factor: ' + factor + ', Score:' + score + '-' * 5 + "\n"
        print(outstring)
        txt_file.write(outstring)
        outstring = "Mean and Variance of {} by {}\n".format(score, factor)
        print(outstring)
        txt_file.write(outstring)
        print(mean_var_table)
        txt_file.write(mean_var_table.to_string())
        
        outstring = "\n>>>ANOVA test for {} by {}\n".format(score, factor)
        print(outstring)
        txt_file.write(outstring)
        group_data = [group for name, group in data_score_factor]
        f_value, p_value = stats.f_oneway(*group_data)
        outstring = "F value: {}, p value: {}\n".format(f_value, p_value)
        print(outstring)
        txt_file.write(outstring)
        if p_value < 0.05:
            outstring = "The mean {} scores of different groups of {} are significantly different, the null hypothesis is rejected.\n".format(score, factor)
            print(outstring)
            txt_file.write(outstring)
        else:
            outstring = "The mean {} scores of different groups of {} are not significantly different, the null hypothesis is not rejected.\n".format(score, factor)
            print(outstring)
            txt_file.write(outstring)
        
        outstring = ">>>Spearman correlation between {} and {}\n".format(factor, score)
        print(outstring)
        txt_file.write(outstring)
        spearman_corr, p_value = stats.spearmanr(df_postcovid[factor], df_postcovid[score])
        outstring = "Spearman correlation : {}, p value: {}\n".format(spearman_corr, p_value)
        print(outstring)
        txt_file.write(outstring)
        if p_value < 0.05:
            outstring = "The correlation between {} and {} is significant, the null hypothesis is rejected.\n".format(factor, score)
            print(outstring)
            txt_file.write(outstring)
        else:
            outstring = "The correlation between {} and {} is not significant, the null hypothesis is not rejected.\n".format(factor, score)
            print(outstring)
            txt_file.write(outstring)
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_postcovid, x=factor, y=score)
        plt.title('Boxplot of {} by {}, post-covid'.format(score, factor))
        plt.xlabel(factor)
        plt.ylabel(score)
        plt.savefig(save_dir + 'postcovid_{}_by_{}.png'.format(score, factor))
        plt.close()


txt_file.close()
