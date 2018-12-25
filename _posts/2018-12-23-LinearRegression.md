---
title: "Basics of Linear Regression, Case Study 1: Medical Costs Analysis"
date: 2018-12-23
tages: [machine learning, data science, linear regression, python]
header:
  image: "/images/LinearRegression/LinReg.jpeg"
excerpt: "Machine Learning, Linear Regression, Data Science"
mathjax: "true"

---

# H1 Heading

## H2 Heading

### H3 Heading

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
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/df.head" alt="First five rows of the dataset">

```python
    df.info()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/df.info alt="Basic information about all columns">

```python
    df.describe()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/df.describe alt="Basic information about all columns">

```python
    df['region'].value_counts()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/df.region alt="Count a number of values for each region">



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

<img src="{{ site.url }}{{ site.baseurl }}/images/LinearRegression/PairPlot.png" alt="Pair plot figure of all data">

Mathematics:

$$ z=x+y $$

Inline putting $$ z=x+y $$
