# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the Cost function using Gradient Descent and generate the required graph.

Program:

## Program:
/*

Program to implement the linear regression using gradient descent.

Developed by: NITHILA.S

RegisterNumber: 212224040224

*/

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```


## Output:

# Data Information
<img width="694" height="268" alt="Screenshot 2025-09-27 103808" src="https://github.com/user-attachments/assets/4166d087-a200-4752-be3e-a0798f0e8595" />

# Value of X
<img width="275" height="884" alt="Screenshot 2025-09-27 103905" src="https://github.com/user-attachments/assets/d4bfd6b9-14e4-4ac6-a1c5-b7d01a658609" />

# Value of X1_Scaled
<img width="425" height="883" alt="Screenshot 2025-09-27 103917" src="https://github.com/user-attachments/assets/e4d48c04-5374-4d1f-9c2c-b4eb4c7cf843" />

# Predicted Value
<img width="299" height="50" alt="Screenshot 2025-09-27 103927" src="https://github.com/user-attachments/assets/5d35eb0f-4a7b-4d7a-8547-9156d5466683" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
