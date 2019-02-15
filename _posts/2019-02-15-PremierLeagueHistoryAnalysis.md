---
title: "Who is the Quarter Century Champion of the Premier League?"
date: 2019-02-15
tags: [pandas, data analysis, data mining, football, python]
excerpt: "Pandas, Data Analysis, DataFrame, Football, Premier League"
mathjax: "true"

---
<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/PL.png" alt="PL logo">

In this chapter, some elemental PANDAS data analysis will be performed, when we are facing ".csv" files in front of us. Dataset of interest contains results from every Premier League match from 1993-1994 to 2017-2018. Columns include Division (denoted as E0), HomeTeam, AwayTeam, FTHG (final time home goals), FTAG (final time away goals), FTR (full time result), HTHG (half time home goals), HTAG (half time away goals), HTR (half time result), and season. The dataset could be obtained from [Premier League dataset](https://www.kaggle.com/thefc17/epl-results-19932018).

The main task in this short chapter will be to announce the Quarter Century Champion of the PL, based on all results recorded in the table. To complete that task, we must do some basic DataFrame transformations.

* For the beginning, we will import two required python libraries and read original csv file and write it into a variable.

```python
    import numpy as np
    import pandas as pd
    data = pd.read_csv("../input/EPL_Set.csv")
```

* Let's see the names of all clubs which played in the PL from 1993 to 2018:

```python
    clubs = data['HomeTeam'].unique()
    clubs
```
<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/Clubs.png" alt="All PL clubs">

* **nunique** function will be used to calculate overall number of clubs which played during the period of 25 last years:

```python
    clubs = data['HomeTeam'].unique()
    clubs
```
<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/NumberClubs.png" alt="Number of clubs">

* If we want to proclaim the champion, the task will be to create a new data frame which will present the number of wins, draws and losses (home and away), for each club from PL.

* Draw games

We will start with extracting draw games from the initial DataFrame.

```python
    Draws = pd.DataFrame()
    Draws = data[(data['FTR']=='D')]
    Draws.head(10)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/DrawGames.png" alt="All draw Games">

Now we want to make two lists, one for presenting numbers of draw games for each club separately (played home), and second one for presenting numbers of draw games for each club (played away).

```python
Draws_Teams_Home = []
    for team in clubs:
        Draws_Teams_Home.append((team,
        Draws[Draws["HomeTeam"] ==team]["Season"].count()))

    Draws_Teams_Away = []
    for team in clubs:
        Draws_Teams_Away.append((team,
        Draws[Draws["AwayTeam"] ==team]["Season"].count()))
```
Let's print only Draws_Teams_Home to see how it looks like.

<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/DrawGamesTeams.png" alt="List of draw games per Team">

Now it is the time to extract all the wins from the table, and than to calculate numbers of wins for each club (home/away)

```python
All_Wins = pd.DataFrame()

All_Wins = data[(data['FTR']=='H') | (data['FTR']=='A')]

Wins_Teams_Home = []
for team in clubs:
    Wins_Teams_Home.append((team,
    All_Wins[(All_Wins["HomeTeam"] ==team)
    & (All_Wins["FTR"] =='H')]["Season"].count()))

Wins_Teams_Away = []
for team in clubs:
    Wins_Teams_Away.append((team,
    All_Wins[(All_Wins["AwayTeam"] ==team)
    & (All_Wins["FTR"] =='A')]["Season"].count()))
```

Further, lists of data which are created previously, will be transformed to a new data frame columns.

```python
    df_home_wins = pd.DataFrame(Wins_Teams_Home,
    columns=['team', 'WinsHome'])

    df_away_wins = pd.DataFrame(Wins_Teams_Away,
    columns=['team2', 'WinsAway'])

    df_home_draws = pd.DataFrame(Draws_Teams_Home,
    columns=['team3', 'DrawsHome'])

    df_away_draws = pd.DataFrame(Draws_Teams_Away,
    columns=['team4', 'DrawsAway'])
```

And let's print one of them to see what we have got.

```python
    df_home_wins
```
<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/WinsHomeDF.png" alt="List of wins home per Team">

We have transformed draw and win games, so the same procedure will be performed with the lost games.

```python
    Losses_Teams_Home = []
    for team in clubs:
        Losses_Teams_Home.append((team,
        All_Wins[(All_Wins["HomeTeam"] ==team)
        & (All_Wins["FTR"] =='A')]["Season"].count()))

    Losses_Teams_Away = []
    for team in clubs:
        Losses_Teams_Away.append((team,
        All_Wins[(All_Wins["AwayTeam"] ==team)
        & (All_Wins["FTR"] =='H')]["Season"].count()))

    df_home_losses = pd.DataFrame(Losses_Teams_Home,
    columns=['team5', 'LossesHome'])
    df_away_losses = pd.DataFrame(Losses_Teams_Away,
    columns=['team6', 'LossesAway'])
```

Now, we have all required information. Let's first create a new DataFrame with only wins and draws.

```python
    Games_Statistic = pd.concat([df_home_wins,df_away_wins,
    df_home_draws,df_away_draws],axis=1)
    Games_Statistic = Games_Statistic.drop(['team2',
    'team3','team4'], axis=1)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/WinsDraws.png" alt="List of wins and draws">

Secondly, information about lost games for each team in a new DataFrame named Games_Summary will be added:

```python
    Games_Summary = pd.concat([Games_Statistic,
    df_home_losses,df_away_losses],axis=1)

    Games_Summary = Games_Summary.drop(['team5', 'team6'], axis=1)
    Games_Summary
```
<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/GamesSummary.png" alt="Games Summary">

New DataFrame is completed. We have a good insight from it about performances of each club from PL.

* Next, we will add a new column with overall number of played games for each club. Also, we will rearrange position of columns in our dataframe. We want the position of the 'number of played games' column to be immediately after the team name column.

```python
    Games_Summary['Played games'] =
    Games_Summary.iloc[:,-7:].sum(axis=1)

    Games_Summary = Games_Summary[['team','Played games',
    'WinsHome','WinsAway','DrawsHome','DrawsAway',
    'LossesHome','LossesAway']]
```

* As it was said at the beginning, the task of this chapter is to calculate the Quarter Century Champion of PL. For finalization of that task, we will create a new column, 'Points', which will be equal to the number of points won for all played games. Of course, every win is equal to 3 points, every draw game will be presented with 1 point.

```python
    Games_Summary['Points'] = (Games_Summary['WinsHome']
    + Games_Summary['WinsAway'])*3 + Games_Summary['DrawsHome']
    + Games_Summary['DrawsAway']

    Games_Summary = Games_Summary.sort_values(by=['Points'],
    ascending = False)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/3 PremierLeague/FinalTable.png" alt="Final Table">

**And we have a champion!!!! Manchester United!!!**

**Interesting fact... In the period 1993 - 2018, Man United have won 2018 league points.**

In the next announcement, we will  try to find some other interesting facts about the Premier League. Of course, **pandas** library will be again our best friend.
