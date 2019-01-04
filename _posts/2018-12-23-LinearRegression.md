---
title: "Basics of Linear Regression, Case Study 1: Medical Costs Analysis"
date: 2018-12-23
tages: [machine learning, data science, linear regression, python]
header:
  image: "/images/LinearRegression/LinReg.png"
excerpt: "Machine Learning, Linear Regression, Data Science"
mathjax: "true"

---

In this example, it will be presented how to use Linear Regression model for the purpose of analyzing the Medical Costs Data. The data is obtained from: https://www.kaggle.com/mirichoi0218/insurance

### First, let's import required libraries

```python
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
```

### Next step is to load the dataset

```python
    df = pd.read_csv("../input/insurance.csv")
```

### The most important thing is to understand the data we have. Let's see some basic information about them.

```python
    df.head()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfHead.png" alt="First five rows of the dataset">

```python
    df.info()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfInfo.png" alt="Basic information about all columns">

```python
    df.describe()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfDescribe.png" alt="Basic information about all columns">

```python
    df['region'].value_counts()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfRegion.png" alt="Count a number of values for each region">


### Next part is to graphically represent our numerical data. We will plot pairplot graph from the seaborn library.

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/PairPlot.png" alt="Pair plot figure of all data">


### The variable of interest is "charges". We want to predict what would be medical costs for specific individual, based on other given information. Let's see distribution of data for "charges". From the figure bellow we can conclude that "charges" parameter is close to normal distribution of data. It is a good thing!

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/distplot.png" alt="Distribution of charges column">



Next figure is a heatmap. It represents mutual correlation of numerical categories from our dataset. Interesting fact - Children category (Number of children covered by health insurance) has the lowest correlation with "charges". Personally, I thought it would be vice versa.

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/heatmap.png" alt="Distribution of charges column">

```python
    df.columns
```

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfColumns.png" alt="Distribution of charges column">


We have three non numerical categories: sex, smoker and region. We want to use them too. So the next step is to convert these variables to numerical values (label values), by which, each string inside a category will be presented with one label (integer).

```python
    from sklearn.preprocessing import LabelEncoder

    lb_sex = LabelEncoder()
    df["sex_label"] = lb_sex.fit_transform(df["sex"])

    lb_smoker = LabelEncoder()
    df["smoker_label"] = lb_sex.fit_transform(df["smoker"])

    lb_region = LabelEncoder()
    df["region_label"] = lb_sex.fit_transform(df["region"])
```

We can see last three columns at dataframe below (which represent converted categories)

```python
    df.head(10)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfHead2.png" alt="Distribution of charges column">


We have prepared our data for further processing. Finally, we can import, initialize and use the Linear Regression model.

```python
    from sklearn.model_selection import train_test_split
```
Five dataframe categories will be used as inputs X for the model. And we want to fit our model according to "charges" category - output Y of the model

```python
    X = df[['age', 'sex_label', 'bmi', 'children', 'smoker_label', 'region_label']]
    y = df['charges']
```

We will use train_test_split function to divide our data to training and testing data.

```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
```

Procedure for importing and fitting the model.

```python
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train,y_train)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/LRmodel.png" alt="Distribution of charges column">



We will create a new dataframe to present estimated coefficientscieints of our model. First one is the intercept, and other coefficients are in correlation with specific categories.

### Interesting fact number 2:###
Quit smoking! We can observe that smoker_label category has the highest influence on increasing medical costs. It has stronger influence than all other analyzed parameters TOGETHER.

```python
    print(lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
    print(coeff_df)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/coeff.png" alt="Distribution of charges column">

Final part is to use fitted model for predicting new values (based on prepared X_test array)

```python
    predictions = lm.predict(X_test)
    print("Predicted medical costs values:", predictions)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/PredictedValues.png" alt="Distribution of charges column">

Graphical comparison of expected values (y_test) and predicted values (predictions).

```python
    plt.scatter(y_test, predictions)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/scatter.png" alt="Distribution of charges column">

Also, let's see error distribution graph of our predictions. Very close to normally distributed data.

```python
    sns.distplot((y_test-predictions), bins=50)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/distplot2.png" alt="Distribution of charges column">

Finally, let's print MAE and MSE erorrs for entire test data.

```python
    from sklearn import metrics
    print(metrics.mean_absolute_error(y_test, predictions))
    print(metrics.mean_squared_error(y_test, predictions))
```

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/metrics.png" alt="Distribution of charges column">
