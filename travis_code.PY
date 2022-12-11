# Import necessities
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# Drop Null values
dataset = dataset.dropna()

# Label Encoding  
le = LabelEncoder()
dataset['lspName'] = le.fit_transform(dataset['lspName'])
dataset['originCity'] = le.fit_transform(dataset['originCity'])
dataset['destinationCity'] = le.fit_transform(dataset['destinationCity'])

# Features setup
X1 = np.asanyarray(dataset[['lspName','originCity','destinationCity']])
y1 = np.asanyarray(dataset[['delayStatus']])
target_names = (['Yes','No'])
feature_names = (['lspName','originCity','destinationCity'])

# Decision tree fit
dt = tree.DecisionTreeClassifier()
dt.fit(X1, y1)

# Prediction
pred = dt.predict(X1)
pred = np.array_str(pred)

# Display Prediction
def predictions(pred):
    if pred=="['Yes']":
        plt.axes()
        res = plt.Rectangle((0,0), 50, 20, fc='red',ec="black")
        plt.gca().add_patch(res)
        plt.axis('off')
        plt.show()
    else:
        plt.axes()
        res = plt.Rectangle((0,0), 50, 20, fc='green',ec="black")
        plt.gca().add_patch(res)
        plt.axis('off')
        plt.show()

predictions(pred)
