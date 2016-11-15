# What is "Connie"
Connie is a fuzzy clustering algorithm which is published under the name "Rederivation of the fuzzy–possibilistic clustering objective function through Bayesian inference". Here is the abstract of the published version of the Connie paper,

> Unsupervised clustering of a set of datums into homogenous groups is a primitive operation required in many signal and image processing applications. In fact, different incarnations and hybrids of Fuzzy C-Means (FCM) and Possibilistic C-means (PCM) have been suggested which address additional requirements such as accepting weighted sets and being robust to the presence of outliers. Nevertheless, arriving at a general framework, which is independent of the datum model and the notion of homogeneity of a particular problem class, is a challenge. However, this process has not been followed organically and clustering algorithms are generally based on exogenous objective functions which are heuristically engineered and are believed to lead to the satisfaction of a required behavior. These techniques also commonly depend on regularization coefficients which are to be set "prudently" by the user or through separate processes. In contrast, in this work, we utilize Bayesian inference and derive a robustified objective function for a fuzzy-possibilistic clustering algorithm by assuming a generic datum model and a generic notion of cluster homogeneity. We utilize this model for the purpose of cluster validity assessment as well. We emphasize the epistemological importance of the theoretical basis on which the developed methodology rests. At the end of this paper, we present experimental results to exhibit the utilization of the developed framework in the context of four different problem classes.

The full version of the Connie paper can be found on [Science Direct](http://www.sciencedirect.com/science/article/pii/S0165011415004947) or on [abadpour.com](http://abadpour.com/files/pdf/Connie_full.pdf). 

# Mathematics of "Connie"

Connie minimizes the following cost function,

[//]: # (\begin{align*}\Delta=\sum_{n=1}^N\omega_n\left[p_n^2\sum_{c=1}^Cf_{nc}^2u_{nc}+(1-p_n)^2UC^{-1}\right]\end{align*})
![Delta](http://quicklatex.com/cache3/39/ql_d165b6f66f4954fd3623fa68113f2c39_l3.png)

subject to,
[//]: # (\begin{align*}\sum_{c=1}^Cf_{nc}=1, \forall n\end{align*})
![sum fnc](http://quicklatex.com/cache3/da/ql_c1f8712f0d547e930e46a1a4716e85da_l3.png)

Connie utilizes a Picard iteration using,

[//]: # (\begin{align*}f_{nc}={{u_{nc}^{-1}}\over{\sum_{c^\prime}^Cu_{nc^\prime}^{-1}}}\end{align*})
![fnc](http://quicklatex.com/cache3/86/ql_426c1608d2912935ead9f22b8cbc9686_l3.png)

[//]: # (\begin{align*}p_n={{1}\over{1+CU^{-1}\sum_{c=1}^Cf_{nc}^2u_{nc}}}\end{align*})
![pn](http://quicklatex.com/cache3/5e/ql_300f946a10b52ef9abbdbec94196745e_l3.png)

Whereas, clusters are updated using the following weights,
[//]: # (\begin{align*}\tilde{\omega}_{nc}=\omega_nf_{nc}^2p_{n}^2u_{nc}^\prime\end{align*})
![wnc](http://quicklatex.com/cache3/e1/ql_3d28dcd16d57ec429d825ffd67a16de1_l3.png)

# Execution of "Connie"
Connie is developed in Python 2.7.6 and is tested in ipython notebook 4.1.2. In order to execute the code call
```
python runme.py
```
or open runme.ipynb in ipython notebook.

# Citation of "Connie"
```
@article{Connie13,
author="Arash Abadpour",
title="Rederivation of the fuzzy–possibilistic clustering objective function through {B}ayesian inference",
journal="Fuzzy Sets and Systems",
volume="305",
number="",
pages="29-53",
year="2016"
}
```

# Why "Connie"?
In "[Jack Goes Boating](https://en.wikipedia.org/wiki/Jack_Goes_Boating_(film))", which is also accidentally [Philip Seymour Hoffman](https://en.wikipedia.org/wiki/Philip_Seymour_Hoffman)'s only work as a director, Jack falls for Connie. 

![Frame from Jack Goes Boating](http://abadpour.com/wp-content/uploads/2016/11/connie_image_small.jpg)

Image is copyright of its owner.

# Acknowledgement
I started working on Connie sometime in 2013, while I was a researcher at Epson Edge. The publication of the paper and the release of this work is courtesy of the management of Epson Edge.

# Rights to "Connie"
This work is open to use by anyone. I understand that this is not a proper description of the legal status of this work and intend to resolve this issue shortly. In the meantime, use this code, don't blow things up as you are using it, and stay tuned.
