import pandas as pd
import numpy as np
import matplotlib as plt

data = pd.read_csv('q2.csv', header = None)
data.columns = ['A', 'B', 'C']
x = data[['A', 'B']]
y = data['C']

# From the matrix form, X'(y - Xb) = 0 and solving for b
xx = x.T.dot(x)
xx_inv = pd.DataFrame(np.linalg.inv(xx), x.columns, x.columns)
beta = xx_inv.dot(x.T).dot(y)

# The fitted values are
y_hat = x.dot(beta)

# The residuals 
e = y - y_hat
