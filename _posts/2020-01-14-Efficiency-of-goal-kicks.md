---
layout: post
title: Efficiency of goal kicks
mathjax: true
category: football
---

## Introduction 
Football is changing all the time. One of the changes is that goalkeepers are functioning as the defenders, they are playing out of the box and starting the plays not only by long balls but also with short passes... and [Pep like it](https://www.manchestereveningnews.co.uk/sport/football/football-news/ederson-man-city-midfield-pep-15680675).

The evolution could be easily seen by looking at the goal kick passes distribution in the Barcelona matches. 
![](https://i.imgur.com/d7tFGLr.png)

In the season 2018/2019 compared with 2004/2005, there were two times fewer 60m passes and 1.5 times more 20m passes. Such a difference is caused by the desire to have more ball control in the build-up phase. 

But is it effective? What is better: 
* [short pass](https://youtu.be/QgAUGWTnt8g?t=198)

<iframe width="420" height="315" src="http://www.youtube.com/embed/QgAUGWTnt8g?t=198" frameborder="0"> </iframe> 

* [midfield lob pass](https://www.youtube.com/watch?v=raNt_hOBeJU&feature=youtu.be&t=34)

<iframe width="420" height="315" src="http://www.youtube.com/embed/raNt_hOBeJU?t=34" frameborder="0"> </iframe> 

* [crazy long ball, which can be a goal itself](https://www.youtube.com/watch?v=TLFpWA0O41Q)

<iframe width="420" height="315" src="http://www.youtube.com/embed/TLFpWA0O41Q" frameborder="0" allowfullscreen> </iframe> 

The main idea of football is to score goals ([but not for everybody](https://www.eurosport.ru/football/russian-premier-league/2019-2020/story_sto7532867.shtml)). To score the goal you need to shoot. Therefore, to estimate the efficiency we calculate the probability for the team to shoot after the goal kick, before losing the possession.

## Methods

**(If you came for football go to the results section directly)**

To answer the question we will be using the [statsbomb data](https://github.com/statsbomb/open-data) of the Barcelona FC seasons from 2004 to 2018. 

Typically the goal kick starts the possession, by losing the same possession. In the best case, the possession in ending by shot. Sometimes it can end with the corner, free kick or penalty shot, but we will skip these cases they are rare (<5%). 

The probabilistic model is step-like, similar to the [PyMC2 tutorial](https://pymc-devs.github.io/pymc/tutorial.html):

$$ \begin{split}     
\begin{array}{ccc}  
(p_{shot} | p_1, p_2, t_1) \sim\text{Bernoulli}\left(p\right), & p=\left\{\begin{array}{lll}             p_1 &\text{if}& l< t_1\\ p_2 &\text{if}& l\ge t_1             \end{array}\right.,&t_1\in[0,1]\\         p_1\sim \text{Uniform}(0, 1)\\         p_2\sim \text{Uniform}(0, 1)\\         t_1\sim \text{Uniform}(0, 1)\\     \end{array}\end{split} $$
 
Shots are described by the Bernoulli distribution. The $p_1$ and $p_2$ is the step function describing the boundary between the short and the long pass. While the boundary itself is defined by the $t_1$ value. All the inputs have no prior information, thus they are initiated uniformly. 

![](https://i.imgur.com/m4MBskS.png)
The example of the 2014/2015 season calculation is presented in the figure.

The algorithm implementation was done in Julia language using [Turing.jl](https://turing.ml/dev/). The inference is calculated by MCMC NUTS sampler.
```julia=
using Turing
import MCMCChains

function step_func(x, thresh, p1, p2)
        if x < thresh
            return p1
        else
            return p2
        end
end

@model bernouli(s, l) = begin
       thresh ~ Uniform(0, 1)
       p1 ~ Uniform(0, 1)
       p2 ~ Uniform(0, 1)
       N = length(s)
       for i = 1:N
           p = step_func(l[i],thresh,p1,p2)
           s[i] ~ Bernoulli(p)
       end
end
   
num_chains = 4
iterations = 2300
burnin = 300
     chain = mapreduce(
             c -> sample(bernouli(shots_dist, length_dist), NUTS(200, 0.65), iterations, progress=true), 
                  MCMCChains.chainscat, 
                  1:num_chains);
#Burn-in
chain = chain[burnin:end,:,:]
# Test chains
gel_test = gelmandiag(chain)
```

Several chains are needed to check the convergence of the MCMC calculation using the [Gelman–Rubin diagnostic](https://arxiv.org/abs/1812.09384). The calculation could be considered converged when the test results are less than $1.1$. 

The calculated probabilities were tested by a 3-step scheme: 
```julia=
using Turing
import MCMCChains

function step3_func(x, thresh, thresh2, p1, p2, p3)
        if x < thresh
            return p1
        else
            if x < thresh2
                return p2
            else
                return p3
            end
        end
end

@model bernouli_3step(s, l) = begin
       thresh ~ Uniform(0, 1)
       thresh2 ~ Uniform(thresh, 1)
       p1 ~ Uniform(0, 1)
       p2 ~ Uniform(0, 1)
       p3 ~ Uniform(0, 1)
       N = length(s)
       for i = 1:N
           p = step3_func(l[i],thresh, thresh2, p1, p2, p3)
           s[i] ~ Bernoulli(p)
       end
end;

   
num_chains = 4
iterations = 2300
burnin = 300
     chain = mapreduce(
             c -> sample(bernouli_3step    (shots_dist, length_dist), NUTS(200, 0.65), iterations, progress=true), 
                  MCMCChains.chainscat, 
                  1:num_chains);
#Burn-in
chain = chain[burnin:end,:,:]
# Test chains
gel_test = gelmandiag(chain)
```
![](https://i.imgur.com/7Qtrr3z.png)

As the results of the 3-step scheme are consistent with the 2-step scheme, we can start making conclusions. 

## Results

The analysis was held to estimate the probability of having a shot after the goal kick. But the threshold value which shows where short passes turn to long passes and can be useful to estimate the 

The 2-steps scheme results are presented below: 
![](https://i.imgur.com/16COkNV.png)

The surprising result that the pass length threshold is fluctuating from one year to another ($\pm$ 30 m), but stays approximately at the same level (14 years trend results in approximately 15 m change). Which means that difference in length between efficient short passes and inefficient long passes had been the same all the time. But managers and analytics started to use it only recently.

The most successful short passes ($p_{shot} > 22\%$) were during very successful Pep Guardiola's season of 2010. La Liga $1^{st}$, Champions League...nothing surprising. 

The largest change happened between 2011/2012 and 2012/2013 season when the Pep Guardiola left the team due to mini-crisis, and Tito Vilanova was in charge. The threshold was equal to 30m and the probability to score after the goal kick was shorter than 30m was three times higher than to score with the long pass. It is showing that Victor Valdez and Tito Vilanova were quite successful in the taming of the build-up by short lob passes ([1](https://www.fourfourtwo.com/performance/skills/victor-valdes-guide-distribution), [2](https://bleacherreport.com/articles/1596674-barcelona-breaking-down-their-formation-and-the-role-of-each-player#slide1)) for one season at least. 

Unfortunately, due to the increasing popularity of the high pressing plays, the threshold is going up to the pre-Guardiola levels. 

**Is there some room for improvement?**

If one takes a look to the 3-step data, then he can spot the surprisingly high efficiency of the ultra-long passes (>80m) in the last two seasons.  
![](https://i.imgur.com/DZ7pMa6.png)
Large error bars due to the rarity of the goalkeepers with such powerful legs, do not allow us to make a strong conclusion. But shooting the ball to the opponent's box could be a viable technique to play against teams which are good in pressing.

![](https://i.imgur.com/STtwYXb.png)

As the conclusion I will separate the goal kick zones to the classes: 
* **Short (l<45m):** Very effective $p_{shot} \approx 13\%$. It is the optimal way to start the possesion, well known. 
* **Long (45m<l<80m):** The less effective $p_{shot} \approx 6\%$, but still popular.
* **Opponent Box (l>80m):** It is the unexplored region with $p_{shot} \approx 8\%$, very rare but could be efficient against pressing teams.

Hope you will find it useful.


## References:

Source code: https://github.com/vsolokha/goalkickanalysis
Pitch plot is inspired by http://petermckeever.com/2019/01/plotting-pitches-in-python/ 
Data provided by https://github.com/statsbomb/open-data
![](https://github.com/statsbomb/open-data/raw/master/stats-bomb-logo.png)
