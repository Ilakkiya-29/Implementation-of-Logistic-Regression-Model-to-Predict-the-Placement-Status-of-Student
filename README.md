# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import required libraries such as NumPy, Pandas, and Scikit-learn.
3. Create or load the dataset containing student details and placement status.
4. Split the dataset into input features (X) and output label (y).
5. Divide the data into training and testing sets.
6. Create a Logistic Regression model.
7. Train the model using the training data.
8. Predict the placement status using test data.
9. Display the accuracy of the model.
10. Stop the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Ilakkiya K
RegisterNumber: 212225040130
*/
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    'CGPA': [6.5, 7.0, 8.2, 5.8, 9.1, 6.0, 8.5, 7.8],
    'Internship': [0, 1, 1, 0, 1, 0, 1, 1],
    'Placement': [0, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['CGPA', 'Internship']]
y = df['Placement']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Scatter plot
plt.figure()
for value in [0, 1]:
    subset = df[df['Placement'] == value]
    plt.scatter(subset['CGPA'], subset['Internship'])

# Decision boundary
x_values = np.linspace(df['CGPA'].min(), df['CGPA'].max(), 100)
y_values = -(model.coef_[0][0] * x_values + model.intercept_[0]) / model.coef_[0][1]

plt.plot(x_values, y_values)
plt.xlabel("CGPA")
plt.ylabel("Internship")
plt.title("Logistic Regression – Placement Status")
plt.show()

```

## Output:
<img width="565" height="453" alt="image" src="https://github.com/user-attachments/assets/3bcdcc36-cbfb-4b20-95d7-54335aa102d5" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
