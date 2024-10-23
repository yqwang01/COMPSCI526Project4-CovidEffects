# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:42:09 2024

@author: 13190
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset_path = './Coursework/COMPSCI526_Data_scicence/Project/Dataset/COVID-19-Constructed-Dataset-(PANEL).xlsx'
df = pd.read_excel(dataset_path)

usable_variable_dict = {
    'numcomputers':[],
    'familysize':[],
    'readingscore':[],
    'writingscore':[],
    'mathscore':[],
    'readingscoreSL':[],
    'writingscoreSL':[],
    'mathscoreSL':[]

}

for index, row in df.iterrows():
    for key in usable_variable_dict:
        usable_variable_dict[key].append(row[key])
#%%
output_score = ['readingscore','writingscore','mathscore','readingscoreSL','writingscoreSL','mathscoreSL']
for output in output_score:

    # Create a DataFrame
    data = pd.DataFrame({
        'Num_Computers': usable_variable_dict['numcomputers'],
        'Family_Size': usable_variable_dict['familysize'],
        'Child_Score': usable_variable_dict[output]
    })

    # Pivot table to calculate average score for each father-mother education pair
    heatmap_data = data.pivot_table(values='Child_Score', 
                                    index='Num_Computers', 
                                    columns='Family_Size', 
                                    aggfunc='mean')

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".1f")

    # Add labels and title
    plt.title(output)
    plt.xlabel('Family Size')
    plt.ylabel('Number of Computers')

    # Display the plot
    plt.show()

#%%


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Sample DataFrame for child scores and father education levels
data = pd.DataFrame({
    'Num_Computers': usable_variable_dict['numcomputers'],
    'Child_Score': usable_variable_dict['readingscore']  # Replace with actual score column
})

# Group by Father Education and calculate mean score
father_avg = data.groupby('Num_Computers')['Child_Score'].mean()

# Create a bar plot for average score by Father Education
plt.figure(figsize=(8, 6))
plt.bar(x=father_avg.index, height=father_avg.values, color='lightblue')
plt.ylim([70, 85])  # Set y-limits





# Set labels and title
plt.xlabel('Number of Computers', fontsize=14)
plt.ylabel('Average Child Score', fontsize=14)
plt.title('Average Child Score by Number of Computers', fontsize=16)

# Remove the top and right spines (borders)
ax = plt.gca()  # Get the current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_yticks([65, 69, 73, 77, 81, 85])  # Define y-ticks below 80
ax.yaxis.set_tick_params(which='both', length=0)  # Remove tick marks
ax.spines['left'].set_bounds(70, 80)  # Keep the bottom spine visible only until 80

# One-Way ANOVA to check for significant differences between groups
anova = ols('Child_Score ~ C(Num_Computers)', data=data).fit()
anova_table = sm.stats.anova_lm(anova, typ=2)
print(anova_table)

# Tukey's HSD test for pairwise comparisons
tukey = pairwise_tukeyhsd(endog=data['Child_Score'], groups=data['Num_Computers'], alpha=0.05)
print(tukey)

# Function to add significance lines between bars
def add_significance_line(x1, x2, y, text):
    """Draws a line between two points with a significance annotation (*)"""
    plt.plot([x1, x2], [y, y], color='black', lw=1.5)  # Line between bars
    plt.text((x1 + x2) * 0.5, y + 0.2, text, ha='center', va='bottom', color='black', fontsize=10)  # * slightly above the line

# Annotate significant differences with lines and asterisks
max_y = father_avg.max() + 1  # Start placing lines a bit above the tallest bar
increment = 0.8  # Define how much higher each line and * should be placed

for i, row in enumerate(tukey._results_table.data[1:]):  # Skip header row
    group1, group2, meandiff, p_adj, lower, upper, reject = row
    if p_adj < 0.05:  # Only annotate significant differences
        # Convert group labels to their corresponding x-axis positions
        x1 = father_avg.index.get_loc(group1)
        x2 = father_avg.index.get_loc(group2)
        
        # Add the significance line and asterisk above the bars
        if p_adj < 0.05 and p_adj>=0.01:
            symbol = '*'
        elif p_adj<0.01 and p_adj>=0.001:
            symbol = '**'
        elif p_adj<0.001:
            symbol='***'
        add_significance_line(x1, x2, max_y, symbol)
        print(p_adj)
        
        # Increase the y-position for the next line to avoid overlap
        max_y += increment

# Show plot
plt.tight_layout()  # Adjust layout to fit the plot nicely
plt.show()

#%%


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Sample DataFrame for child scores and father education levels
data = pd.DataFrame({
    'Family_Size': usable_variable_dict['familysize'],
    'Child_Score': usable_variable_dict['readingscore']  # Replace with actual score column
})

# Group by Father Education and calculate mean score
mother_avg = data.groupby('Family_Size')['Child_Score'].mean()

# Create a bar plot for average score by Father Education
plt.figure(figsize=(8, 6))
plt.bar(x=mother_avg.index, height=mother_avg.values, color='pink')
plt.ylim([70, 85])  # Set y-limits





# Set labels and title
plt.xlabel('Family Size', fontsize=14)
plt.ylabel('Average Child Score', fontsize=14)
plt.title('Average Child Score by Family Size', fontsize=16)

# Remove the top and right spines (borders)
ax = plt.gca()  # Get the current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_yticks([65, 69, 73, 77, 81, 85])  # Define y-ticks below 80
ax.yaxis.set_tick_params(which='both', length=0)  # Remove tick marks
ax.spines['left'].set_bounds(70, 80)  # Keep the bottom spine visible only until 80

# One-Way ANOVA to check for significant differences between groups
anova = ols('Child_Score ~ C(Family_Size)', data=data).fit()
anova_table = sm.stats.anova_lm(anova, typ=2)
print(anova_table)

# Tukey's HSD test for pairwise comparisons
tukey = pairwise_tukeyhsd(endog=data['Child_Score'], groups=data['Family_Size'], alpha=0.05)
print(tukey)

# Function to add significance lines between bars
def add_significance_line(x1, x2, y, text):
    """Draws a line between two points with a significance annotation (*)"""
    plt.plot([x1, x2], [y, y], color='black', lw=1.5)  # Line between bars
    plt.text((x1 + x2) * 0.5, y + 0.2, text, ha='center', va='bottom', color='black', fontsize=10)  # * slightly above the line

# Annotate significant differences with lines and asterisks
max_y = mother_avg.max() + 1  # Start placing lines a bit above the tallest bar
increment = 0.8  # Define how much higher each line and * should be placed

for i, row in enumerate(tukey._results_table.data[1:]):  # Skip header row
    group1, group2, meandiff, p_adj, lower, upper, reject = row
    if p_adj < 0.05:  # Only annotate significant differences
        # Convert group labels to their corresponding x-axis positions
        x1 = mother_avg.index.get_loc(group1)
        x2 = mother_avg.index.get_loc(group2)
        
        # Add the significance line and asterisk above the bars
        if p_adj < 0.05 and p_adj>=0.01:
            symbol = '*'
        elif p_adj<0.01 and p_adj>=0.001:
            symbol = '**'
        elif p_adj<0.001:
            symbol='***'
        add_significance_line(x1, x2, max_y, symbol)
        print(p_adj)
        
        # Increase the y-position for the next line to avoid overlap
        max_y += increment

# Show plot
plt.tight_layout()  # Adjust layout to fit the plot nicely
plt.show()

