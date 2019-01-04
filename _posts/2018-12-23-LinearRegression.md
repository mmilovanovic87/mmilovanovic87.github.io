---
title: "Basics of Linear Regression, Case Study 1: Medical Costs Analysis"
date: 2018-12-23
tages: [machine learning, data science, linear regression, python]
header:
  image: "/images/LinearRegression/LinReg.png"
excerpt: "Machine Learning, Linear Regression, Data Science"
mathjax: "true"

---
This will be my first announcement. For the beginning, let's see how to use Python and Linear Regression model to predict data. In this example, it will be presented how to simply analyze a raw data and to use regression model for the purpose of analyzing the Medical Costs Data. The data is obtained from: *https://www.kaggle.com/mirichoi0218/insurance*.

* First, let's import required libraries for further work.

```python
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
```

* Next step is to load the dataset.

```python
    df = pd.read_csv("../input/insurance.csv")
```

*  The most important thing at the beginning is to understand the data. There is no point in building machine learning models if we do not understand what are we looking for in our data and what are our expectations. Let's see some basic information about data set.

+ What categories are included in our data set?

```python
    df.columns
```
  <img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfColumns.png" alt="Distribution of charges column">

+ We will print first five rows of our DataFrame.

```python
    df.head()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfHead.png" alt="First five rows of the dataset">

  As we can see, 7 categories are included through the data.

+ Next, number of entries for each category and column types are presented. Four numerical and three object categories are incldued.

```python
    df.info()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfInfo.png" alt="Basic information about all columns">

+ Describe function will show  characteristics of all numerical categories.

```python
    df.describe()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfDescribe.png" alt="Basic information about all columns">

+ As it is said, three object categories are included. They are categorical features, and the easiest way to analyze these data is to a number of impressions of each possible outcome. For example, region category could be summed as:

```python
    df['region'].value_counts()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfRegion.png" alt="Count a number of values for each region">


* Next part is to graphically represent our numerical data. We will plot pairplot graph from the seaborn library.

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/PairPlot.png" alt="Pair plot figure of all data">

Pairplot is the easiest way to see all mutual correlations and distributions of data, for all numerical categories.

* Let's now suppose that the variable of interest is "charges". We want to predict what would be medical costs for specific individual, based on other given information. Let's see distribution of data for "charges".

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/distplot.png" alt="Distribution of charges column">

From the figure above we can conclude that distribution of "charges" is close enough to normal distribution. It is a good thing!

* Next figure is a heatmap. It represents mutual correlation of numerical categories from our dataset.

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/heatmap.png" alt="Heatmap">

And we have found 1st Interesting fact: Children category (Number of children covered by health insurance) has the lowest correlation with "charges". Personally, I thought it would be vice versa.

* We have three non numerical categories: sex, smoker and region. We want to use them too for the prediction purposes. So the next step is to convert these variables to numerical dummy variables (label values), by which, each string inside a category will be presented with one label (integer).

```python
    from sklearn.preprocessing import LabelEncoder

    lb_sex = LabelEncoder()
    df["sex_label"] = lb_sex.fit_transform(df["sex"])

    lb_smoker = LabelEncoder()
    df["smoker_label"] = lb_sex.fit_transform(df["smoker"])

    lb_region = LabelEncoder()
    df["region_label"] = lb_sex.fit_transform(df["region"])
```

* In the table below we can recognize three new columns (which represent converted object categories)

```python
    df.head(10)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/dfHead2.png" alt="Head of data">

Now we can use all 7 categories for prediction purposes.

* We have prepared our data for processing and fitting a new model. Next part is to use function *train_test_split* to divide all information in two subsets: training and testing subsets.

```python
    from sklearn.model_selection import train_test_split
```
+ Five dataframe categories will be used as inputs of the model. We want to fit our model according to "charges" category, so output variable Y will be "charges" of course.

```python
    X = df[['age', 'sex_label', 'bmi', 'children', 'smoker_label', 'region_label']]
    y = df['charges']
```

+ Usage of *train_test_split* function

```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
```

* Finally, we can proceed with the procedure of importing and fitting the model of Linear Regression.

```python
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train,y_train)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/LRmodel.png" alt="LR model">

* The model is fitted. We will create a new dataframe to present estimated coefficients obtained by our model. Each coefficient represents single category. First number, above the coefficients, is the intercept term of linear regression.

```python
    print(lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
    print(coeff_df)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/coeff.png" alt="Coefficients of LR model">

* Interesting fact 2: Quit smoking! We can observe that smoker_label category from the table above has the highest influence on increasing medical costs. It has stronger influence than all other analyzed parameters TOGETHER.

* Final part of this short case study is to test trained model for predicting new values (by using prepared X_test array which is obtained by train_test_split function)

```python
    predictions = lm.predict(X_test)
    print("Predicted medical costs values:", predictions)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/PredictedValues.png" alt="Predicted values">

+ Graphical comparison of expected values (y_test) and obtained predicted values (predictions).

```python
    plt.scatter(y_test, predictions)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/scatter.png" alt="Scatter graph">

+ Also, let's see an error distribution graph of our predictions. Very close to normally distributed data.

```python
    sns.distplot((y_test-predictions), bins=50)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/distplot2.png" alt="Distplot 2">

+ Finally, let's print mean absolute error (MAE) and mean squared error (MSE) for our predictions

```python
    from sklearn import metrics
    print(metrics.mean_absolute_error(y_test, predictions))
    print(metrics.mean_squared_error(y_test, predictions))
```

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/metrics.png" alt="Metrics">
