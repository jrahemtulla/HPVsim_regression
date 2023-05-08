import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('nigeria_screening_results_massive_sim.xlsx')


#split scen_label column based on screening prob, treatment prob, sens/ spec
new_columns = df['scen_label'].str.split('[,/_]', expand=True)

#rename columns of the resulting data frame
new_columns.columns = ['product','screening_prob','treatment_prob','sensitivity','specificity']

# Add the new columns to the original DataFrame
df[new_columns.columns] = new_columns
df = df.rename(columns={'scen_label': 'original_scen_label'})

# Remove the '%' symbol from the column
df['sensitivity'] = df['sensitivity'].str.replace('%', '')
df['specificity'] = df['specificity'].str.replace('%', '')

# Convert the column from string to float
df['sensitivity'] = df['sensitivity'].astype(float)/100
df['specificity'] = df['specificity'].astype(float)/100
df['treatment_prob'] = df['treatment_prob'].astype(float)
df['screening_prob'] = df['screening_prob'].astype(float)
df['cancer_deaths'] = df['cancer_deaths'].astype(float)

#select relevant columns only and data for 2060, remove the nan value from no screening
df = df[df['year'] == 2060].loc[:, ['year', 'asr_cancer_incidence', 'treatment_prob', 'screening_prob', 'sensitivity', 'specificity', 'cancer_deaths']].dropna(subset=['screening_prob'])

# Print the resulting dataframe
print(df.dtypes)
print(df)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.3, random_state=0)

# Create the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the class labels for the testing data
y_pred = model.predict(X_test)

# Get the R-squared value
r_squared = r2_score(Y, y_pred)

# Get the MSE
mse = mean_squared_error(Y, y_pred)

#  Print the intercept and coefficients of the regression model
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("r_squared:", r_squared)

# Evaluate the performance of the model
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:", confusion_matrix(y_test, y_pred))