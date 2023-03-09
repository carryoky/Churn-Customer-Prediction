# Predict Customer Churn in Python
Customer attrition is one of the biggest expenditures of any organization. If we could figure out why a customer leaves and when they leave with reasonable accuracy, it would immensely help the organization to strategize their retention initiatives manifold. 
By the end of this,let’s attempt to solve some of the key business challenges pertaining to customer attrition like say, (1) what is the likelihood of an active customer leaving an organization? (2) what are key indicators of a customer churn? (3) what retention strategies can be implemented based on the results to diminish prospective customer churn?
In real-world, we need to go through seven major stages to successfully predict customer churn:
*Section A: Data Preprocessing
*Section B: Data Evaluation

## Section A: Data Preprocessing
### Step 1: Import relevant libraries:
Import all the relevant python libraries for building supervised machine learning algorithms.
```python
import numpy as np                # linear algebra
import pandas as pd               # data processing, read CSV file   
import seaborn as sns             # data visualization
import matplotlib.pyplot as plt   # calculate plots
import missingno as msno          # calculate missing value          
import cufflinks as cf
import plotly.graph_objects as g
```
### Step 2: Import the dataset:
```python
df = pd.read_csv('/home/sam/Downloads/Churn_Modelling.csv')
```
### Step 3: Evaluate data structure:
```python
df.shape
df.columns.values
df.info()
df.describe()
df.columns.to_series().groupby(df.dtypes).groups
df.dtypes
``` 
## Section B: Data Evaluation
### Exploratory Data Analysis:
Let’s try to explore and visualize our data set by doing distribution of independent variables to better understand the patterns in the data and to potentially form some hypothesis.
* Plot countplot of numeric Columns:
```python
sns.countplot(x='Exited', hue='Geography', data=df, palette='husl')
plt.show()
```
* Plot the boxplot of the data:
```python
sns.countplot(x='IsActiveMember', data=df)
plt.show()
sns.boxplot(x='Exited', y='IsActiveMember', data=df)
plt.show()
```
