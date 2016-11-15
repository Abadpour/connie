# What is "Connie"
Connie is a fuzzy clustering algorithm which is published under the name "Rederivation of the fuzzy–possibilistic clustering objective function through Bayesian inference". Here is the abstract of the published version of the Connie paper,

> Unsupervised clustering of a set of datums into homogenous groups is a primitive operation required in many signal and image processing applications. In fact, different incarnations and hybrids of Fuzzy C-Means (FCM) and Possibilistic C-means (PCM) have been suggested which address additional requirements such as accepting weighted sets and being robust to the presence of outliers. Nevertheless, arriving at a general framework, which is independent of the datum model and the notion of homogeneity of a particular problem class, is a challenge. However, this process has not been followed organically and clustering algorithms are generally based on exogenous objective functions which are heuristically engineered and are believed to lead to the satisfaction of a required behavior. These techniques also commonly depend on regularization coefficients which are to be set "prudently" by the user or through separate processes. In contrast, in this work, we utilize Bayesian inference and derive a robustified objective function for a fuzzy-possibilistic clustering algorithm by assuming a generic datum model and a generic notion of cluster homogeneity. We utilize this model for the purpose of cluster validity assessment as well. We emphasize the epistemological importance of the theoretical basis on which the developed methodology rests. At the end of this paper, we present experimental results to exhibit the utilization of the developed framework in the context of four different problem classes.

The full version of the Connie paper can be found on [Science Direct](http://www.sciencedirect.com/science/article/pii/S0165011415004947) or on [abadpour.com](http://abadpour.com/files/pdf/Connie_full.pdf). 


# Execution
Connie is developed in Python 2.7.6 and is tested in ipython notebook 4.1.2. In order to execute the code call
```
python runme.py
```
or open runme.ipynb in ipython notebook.

# Citation
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

# Rights
This work is open to use by anyone. I understand that this is not a proper description of the legal status of this work and intend to resolve this issue shortly. In the meantime, use this code, don't blow things up as you are using it, and stay tuned for the update.
