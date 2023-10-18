# Climate_forcasting
﻿
 
## 1. Intoduction

This is a class project I made aiming to solve the [Regional CLimate Forecasting challenge](https://challengedata.ens.fr/participants/challenges/80/) on challengedata. I've ended up with 4th place on the public scoreboard and 1st place on the private one. Here I will give the idea of the solution and possible ways to improve my solution.

<a name="Challenge"/>
 
## 2. Brief Introduction to the challenge

The goal is to predict regional temperature anomaly given the data of the last 10 years together with the 11-year predictions of other 22 models.

The regions are given by the nested Healpix(one may check [link](https://en.wikipedia.org/wiki/HEALPix) for the disctription of Healpix), which is a way to partition the global into smaller zones. Here we will partition into 3072 regions.

<a name="Goal"/>
 
## 3. Goal of the challenge

The input data $X$ is then given for each zone $k\in$ {0, ..., 3071} a set of $22\times 11+10=252$ variables denoted by $MY_{i, j}$, where $i\in$ {0, ..., 22} corresponds to the index of the model(0 will be the observation) and $j$ between 9 or 10 depending on the value of $i$. The aim to predict the average temperature of the scale-down region, which will be the consecutive 16 points of in the original division.

The target $Y$ will be the average temperature of the scale-down region, which is each of the 16 consecutive zones in the original division.

The output should have the same number of rows as of $Y$, and each row with two values: mean and variance. Which corresponds to the mean and the variance of each scale-down region.

The goal is thus to predict $Y$ given $X$ so that both the prediction is close and the variance is low.

<a name="Mark"/>
 
## 4. Benchmark

The benchmark of this challenge is the loss obtained by taking the last observed data $MY_{0, 9}$ as prediction which is -1.139 on the public dataset and -1.212 on the private data set.

<a name="Idea"/>
 
## 5. Idea

Since the dimension is way too large to be able to treated directly, the first task is to try to reduce the dimensionality. 

Directly applying PCA will lead to more than 50 variables to reach 80 percent of variance explained. Applying direct variable selection will lead to some weird combinations of variables. 

My idea of encountering the problem above is quite simple, I will try to set a criterion to find the right models to take into account. THe criterion I used is to consider the mean of each model, and use that as a prediction to calculate the loss with the target. Then we will choose those that perform better. 

To do this, we will calculate the loss seperately for each dataset, so that we will see which models perform better on which dataset. Fortunately, models 8, 0, 5, 1, 17 pop out on the top on basically every dataset, this gives a starting point to continue with the construction of models.

Unfortunately, after some tries I cannot improve the results better than the pure mean of the models 0, 5, 8. 

<a name="Rank"/>
 
## 6. Ranking

On public ranking, here we only consider the data of models 0, 5, 8
<ul>
<li>Mean scored -1.014</li>
<li>Linear Regression with PCA scored -1.061</li>
<li>Linear Regression without PCA scored</li>
</ul>
On private ranking,
<ul>
 <li>Mean scored -0.750</li>
 <li>Linear Regression with PCA scored -0.761</li>
</ul>

<a name="Improv"/>
 
## 7. Possible Improvements

I believe there should be a lot to improve on my current result. 
<ul>
<li>Try a different criterion for the selection of the models.</li>
<li>Use other methods with the variables obtained.</li>
<li>Use the probabilistic forecasting method introduced in https://www.nature.com/articles/s41467-018-05442-8.</li>
</ul>
