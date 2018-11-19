# LAB 3



## Assignment 1

Firstly, we use the provided function genBlobs(), to plot the the Gaussian distributed datapoints, together with their class labels, providing a number of centers as parameters. Then we add the ML estimates, and plot it all together with the function plotGaussian().

![a1_1](imgs/a1_1.png)

*Figure 1.1 - centers = 5*



## Assignment 2

The following functions were completed:

- computePrior(labels)
- classifyBayes(X,prior,mu,sigma)



## Assignment 3

### Iris dataset

The following is the datapoints including blobs:

![Figure_6](imgs/plotgaussian.png)

The following is the plot boundary:

![Figure_7](imgs/plotboundary.png)

Trial | Accuracy
--- | ---
0 | 84.4
10 | 86.7
20 | 88.9
30 | 86.7
40 | 86.7
50 | 91.1
60 | 88.9
70 | 93.3
80 | 86.7
90 | 88.9

*Final mean classification accuracy: 88.1*  
*Standard deviation: 4.14*

### Vowel dataset
Trial | Accuracy
--- | ---
0 | 58.4
10 | 66.9
20 | 67.5
30 | 57.8
40 | 63.6
50 | 63
60 | 64.9
70 | 62.3
80 | 59.1
90 | 70.1

*Final mean classification accuracy: 62.6*  
*Standard deviation: 3.54*

1) **When can a feature independence assumption be reasonable and when not?**
In general, it is often not realistic to think features are independent, but it may be a useful assumption in order to reduce complexity. For example, if we want to classify  book genres, we can use the words as features. The words will often indicate the genre of book, but occurence of words may also correlate to occurence of other words. However, this factor is very hard to actually mathematically implement, and the model may still be very powerful regardsless of feature dependence, so the assumption can be reasonable to make. However, it is not reasonable to made such an assumption when its probable that there is a very high dependence of features or when there is a high risk involved in a bad model.

2) **How does the decision boundary look for the Iris dataset? How could one improve the classication results for this scenario by changing classifier or, alternatively, manipulating the data?**
There are two boundaries which divide the three classes. The classes 0 and 1 are divided successfully, but the classes 1 and 2 are divided using a quite unintuitive curve. By looking at the datapoints, it should rather be divided with a straight line leaning up-right.A low-complexity SVM with some slack would probably be a good alternative of a classifier.



## Assignment 4

```python
[1] weightsum = sum(W[idx,:])
[2] featuresWeighed = X[idx,:]*W[idx,:] #get row vectors
[3] mu[jdx] += np.sum(featuresWeighed, axis=0) / weightsum
```

[2] We take each datapoint's features and multiply them with the weight for the datapoint.

[3] We look at each dimension one at a time. And for each dimension, we add up all the featuresWeighed i.e. N number of datapoints for that dimension and dataset.



## Assignment 5

```python
testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)

   Trial: 0 Accuracy 100
   Trial: 10 Accuracy 97.8
   Trial: 20 Accuracy 93.3
   Trial: 30 Accuracy 93.3
   Trial: 40 Accuracy 97.8
   Trial: 50 Accuracy 86.7
   Trial: 60 Accuracy 93.3
   Trial: 70 Accuracy 95.6
   Trial: 80 Accuracy 93.3
   Trial: 90 Accuracy 95.6
  
Final mean classification accuracy  94.2 with standard deviation 6.83
```

```python
testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)

    Trial: 0 Accuracy 68.8
    Trial: 10 Accuracy 77.9
    Trial: 20 Accuracy 77.3
    Trial: 30 Accuracy 70.1
    Trial: 40 Accuracy 68.8
    Trial: 50 Accuracy 68.2
    Trial: 60 Accuracy 77.3
    Trial: 70 Accuracy 70.8
    Trial: 80 Accuracy 71.4
    Trial: 90 Accuracy 82.5
	
Final mean classification accuracy  74.1 with standard deviation 3.73
```

![a4_1](imgs/a4_1.png)

