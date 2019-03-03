# Twitter-Sentiment-Analysis
**Problem Statement:** 

>Given a collection of tweets, the program classify them into two classes namely:
- Positive
- Negative
- Neutral (avoided)
- Mixed (avoided)    

**Dataset Description:**

>The tweets used here are pertaining to two US presidential candidates namely: Barack Obama and Mitt Romney. By classifying the tweets into the mentioned classes we would be capable of predicting the opinion of the public and get a sense of the outcome of the election.

**Classification Methods:**
1. MultinomialNaiveBayesClassifier
2. Support Vector Machines(SVM) with RBF/Gaussian Kernel
3. Stochastic Gradient Descent(SGD)
4. LogisticRegression
5. Ensemble Methods:
      - Random Forest
      - Bagging
      - Boosting(XG-Boost)
      - Voting using SVM with Gaussian kernel, Logistic Regression, Random Forest and Stochastic Gradient Descent.
      - Neural Network: Convolutional Neural Network

**Evaluation**
>ExperimentalResults
1. Obama
![Result Table for Obama](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)

**References**
1. http://nltk.org/ - for documentation about NLTK libraries
2. Bing Liu. “Sentiment Analysis and Opinion Mining” , May 2012. eBook: ISBN
      9781608458851
3. http://streamhacker.com/2010/10/25/training-binary-text-classifiers-nltk-trainer/ - for
info on text classification methods
4. http://www.ravikiranj.net/drupal/201205/code/machine-learning/how-build-twitter-
sentiment-analyzer - for information on tweet classification
5. http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
6. http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-
tensorflow/
7. https://keras.io/
8. https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
9. https://blog.statsbot.co/text-classifier-algorithms-in-machine-learning-acc115293278
10. https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6
11. https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-
1.html
