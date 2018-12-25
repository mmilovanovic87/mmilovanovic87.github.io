---
title: "Basics of Linear Regression, Case Study 1: Medical Costs Analysis"
date: 2018-12-23
tages: [machine learning, data science, linear regression, python]
header:
  image: "/images/LinearRegression/LinReg.jpeg"
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




And *italic text*

And some **bold** text

Bulleted list:

* First thing
* Second thing
- Bla bla thing

A numbered list:

1. First
2. seconds




Here is a code 'x+y=c'

Here is an image:



Mathematics:

$$ z=x+y $$

Inline putting $$ z=x+y $$
