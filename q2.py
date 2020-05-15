import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('q2.csv', header = None)
data.columns = ['A', 'B', 'C']

# Lets visualize the data:
data[['A','B']].plot()
plt.show()

data['C'].plot()
plt.show()

sns.pairplot(data)
plt.show()

# Just by looking at the data, we can see that there is a large event around the 200th 
# entry and after that, the variable y is more volatile
# we have no information that hints that this would be a time series model
y = data.C
y_mean = y.mean()
y_std = y.std()
y_normalized = (y - y_mean) / y_std

outlier = y[np.abs(y_normalized) > 3].index
data_clean = data.copy()
data_clean.loc[outlier] = np.nan
data_clean = data_clean.dropna()

sns.pairplot(data_clean)
plt.show()

# After removing the outlier, we can see a cuadratic relationship in the data. 
We look into different interaction terms
x = data_clean[['A', 'B']] 
y = data_clean['C']

# Add intercept and other transformations (iteratively)
x['intercept'] = 1
x['a2b'] = x.A**2 * x.B

# Get the betas using the matrix form, X'(y - Xb) = 0 and solving for b
xx = x.T.dot(x)
xx_inv = pd.DataFrame(np.linalg.inv(xx), x.columns, x.columns)
beta = xx_inv.dot(x.T).dot(y)

# The fitted values are
y_hat = x.dot(beta)

# Calculate the residuals and the standard errors for the coef
e = y - y_hat

# Plot the residuals
e.plot()
plt.title('Residuals')
plt.show()

# Calculate standard errors for the coefficients
ssq = sum(e**2) / (x.shape[0] - x.shape[1])
var_beta = ssq * xx_inv
se_beta = np.diag(var_beta ** 0.5)

# We have heteroskedasticity, let's calculate HAC Standard Errors
e2_matrix = pd.DataFrame(np.diag(e**2), index = x.index, columns = x.index)
var_beta_hac =  xx_inv.dot(x.T).dot(e2_matrix).dot(x).dot(xx_inv)
se_hac = np.diag(var_beta_hac ** 0.5)

# Result
result = pd.DataFrame({'Betas':beta, 'HAC SE_betas':se_hac, 'T-Stat': beta / se_beta})
print(result)

# Plot
plt.scatter(y, y_hat)
plt.title('Y vs Y Fitted')
plt.show()