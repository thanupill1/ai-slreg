import numpy as np #library designed for numerical and scientific computing.
import matplotlib.pyplot as plt # Simple and intuitive to create visuals by combining functions for plotting
import pandas as pd # Library widely used for data manipulation, analysis, and cleaning.
from sklearn.model_selection import train_test_split as tts # library for splitting data, tuning hyperparameters. 
from sklearn.linear_model import LinearRegression # Linear Regression for simple and multiple regression problems.
from sklearn.metrics import r2_score as rscr # Library for is essential for evaluating model performance. 
import pickle # Library for save and load specific objects!
import warnings # Suppress the warning 

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

regr = LinearRegression() # Assign function to a variable.
# Load dataset
data = pd.read_csv(r'Salary_Data.csv') # Load input value into memory

# Independent and dependent variables
indep = data[["YearsExperience"]] # Loar independent value into variable
dep = data[["Salary"]] # Loar dependent value into variable

# Scatter plot
# plt.scatter(indep, dep)
# plt.xlabel("YearsExperience", fontsize=20)
# plt.ylabel("Salary", fontsize=20)
# plt.show()  # Uncommented to show the plot

# Split the data into training and testing sets 
x_train, x_test, y_train, y_test = tts(indep, dep, test_size=1/3, random_state=0)

# Train the model
try:
    regr.fit(x_train, y_train)
except:
    pass #ignore error

# Get the model's weights (coefficients) and intercept
weight = regr.coef_
bais = regr.intercept_
print("Weight of the model = ", weight)
print("Intercept of the model = ", bais)

# Predict using the trained model
y_pred = regr.predict(x_test)

# Calculate and print R-squared score
rscore = rscr(y_test, y_pred)
print(f"Prediction accuracy: {rscore*100:.2f} %")

#if Good model export the file 
if (rscore > 0.95):
    fn = "plan1-final.sav"
    pickle.dump(regr,open(fn,'wb'))
    #import the file for testing 
    lmlp  = pickle.load(open(fn,"rb"))
    res15 = lmlp.predict([[15]])
    res13 = lmlp.predict([[13]])
    print ("Good Model")
    print()
    print()
    print("* * * Two Test for the imported model * * *")
    for i in range(2):
        yoexp = int(input("Enter the candidates' years of experience? ")) # getting input for the test data 
        respo = lmlp.predict([[yoexp]]) # calling imorted module
        print(f"Projected salary for candidates with {yoexp} years of experiencey = {float(respo[0][0]):,.2f}") # Display the response 
        print()
else: 
    print ("Bad Model")


