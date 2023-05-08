import pandas as pd
import sciris as sc
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

#Convert objfile from massive_sim to excel
#screening = sc.loadobj('nigeria_screening_results_massive_sim.obj')
#screening.to_excel('nigeria_screening_results_massive_sim.xlsx')


#load in excel file of simulations (projected data)
df = pd.read_excel('nigeria_screening_results_massive_sim.xlsx')

#Extract relevant data from massive_sim file and set it up for regression

#split scen_label column based on screening prob, treatment prob, sens/ spec
#this will create separate columns of the metadata of each simulation

new_columns = df['scen_label'].str.split('[,/_]', expand=True)

#rename columns of the resulting data frame
new_columns.columns = ['product','screening_prob','treatment_prob','sensitivity','specificity']

# Add the new columns to the original DataFrame
df[new_columns.columns] = new_columns
df = df.rename(columns={'scen_label': 'original_scen_label'})

# Remove the '%' symbol from the column
df['sensitivity'] = df['sensitivity'].str.replace('%', '')
df['specificity'] = df['specificity'].str.replace('%', '')

# Convert the columns from strings to floats, and adjust the sensitivity and specificity to be in number not % form
df['sensitivity'] = df['sensitivity'].astype(float)/100
df['specificity'] = df['specificity'].astype(float)/100
df['treatment_prob'] = df['treatment_prob'].astype(float)
df['screening_prob'] = df['screening_prob'].astype(float)
df['cancer_deaths'] = df['cancer_deaths'].astype(float)

#select data for 2060 only and include only relevant columns, remove the nan value from no screening (i.e. the simulation where no screening is performed)
df = df[df['year'] == 2060].loc[:, ['year', 'asr_cancer_incidence', 'treatment_prob', 'screening_prob', 'sensitivity', 'specificity', 'cancer_deaths']].dropna(subset=['screening_prob'])

# Print the resulting dataframe to check that the data was appropriately cleaned
print(df.dtypes)
print(df)

dep_variable = [
    #'asir',
    'cancer_deaths',
]

if 'asir' in dep_variable:

# Build linear regression model to determine the impact of screening prob, treatment prob, sens and spec on ASIR

    X= df[['screening_prob','treatment_prob','sensitivity','specificity']]
    y= df['asr_cancer_incidence']

if 'cancer_deaths' in dep_variable:
    X= df[['screening_prob','treatment_prob','sensitivity','specificity']]
    y = df['cancer_deaths']

# add a constant term to the predictor variables
    X = sm.add_constant(X)

# fit the multiple linear regression model
    model = sm.OLS(y, X).fit()

# print the summary statistics
    print(model.summary())





