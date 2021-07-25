# IRIS
Iris dataset is a multivariate dataset introduced by British statistician and biologist Ronald Fisher 
in his 1936 paper. It includes three iris species (Iris setosa, Iris Virginica, Iris versicolor) with 50 
samples each as well as some properties about each flower. One flower species is linearly 
separable from the other two, but the other two are not linearly separable from each other.

Attribute Information is given below:

  1. Sepal length in cm
  2. Sepal width in cm
  3. Petal length in cm
  4. Petal width in cm
  5. Class: (Iris Setosa , Iris Virginica, Iris Versicolor)
  
#### OBJECTIVE
Download Iris dataset from the UCI repository and compare the results of data 
analytic techniques. Here Decision tree ,Random Forest, Naïve Bayes, Logistic regression techniques of 
Classification are used for comparison.
#### IMPORTING THE DATASET

  `iris<- read.csv(“c:/iris/iris.csv”)`
#### PREVIEW OF DATASET

 `View(iris) #view dataset` <br>
 `str(iris) #view structure of dataset` <br>
 `summary(iris) #view statistical summary of dataset` <br>
 `head(iris) #view top 6 rows of dataset` <br>
#### DATA VISUALIZATION
A visual representation of how the data points are distributed with respect to the frequency.
Analysis with the histogram:

![visualization](https://github.com/Athira-M-Chandran/Images/blob/main/iris_visualization.jpeg?raw=true)

* The distribution of Iris-Setosa petal is completely different from other 2 species
* The species can’t be separated from one another using sepal features since the 
distribution is overlapping.
* Petal length and petal width can be used as a factor to identify 3 species

#### SPLITTING OF DATA
To increase the adaptability of the model, the entire data is divided into "train_data" and 
"test_data" sets. ‘caTools’ package is used for sample.split() <br>
`sample_data = sample.split(iris, SplitRatio = 0.75) # splits the data in the ratio mentioned in SplitRatio`<br>
`train_data <- subset(iris, sample_data == TRUE) # a training dataset which are marked as TRUE` <br>
`test_data <- subset(iris, sample_data == FALSE) # a testing dataset which are marked as FALSE` <br>
#### DATA ANALYTICS TECHNIQUES
### 1. DECISION TREE

Decision tree is a type of supervised learning algorithm mostly used for classification problem. 
This algorithm split the data into two or more homogeneous sets based on the most significant 
attributes making the group as distinct as possible.<br>
In R, rpart is for modeling decision trees and rpart.plot package enables the plotting of a tree. 
To predict which factors such as sepal length, sepal width , petal length, petal width determine 
the species of iris flower.
  
`fit<- rpart(class~ ., method = "class", data = train_data,  control = rpart.control(cp = 0), parms = list(split="information"))`<br>
`rpart.plot(fit,type= 4 , extra=1)`<br>
![Decision tree](https://github.com/Athira-M-Chandran/Images/blob/main/iris_tree.jpeg?raw=true) <br>
Checking the accuracy using a confusion matrix by comparing predictions to actual 
classifications. ‘caret’ package is used for confusion matrix.
 
`iris_pred <- predict(object = fit, newdata = test_data, type = "class") #test data is used for prediction`<br>
`confusionMatrix(data = as.factor(iris_pred), reference = as.factor(test_data$class))`

![confusion matrix of decision tree](https://github.com/Athira-M-Chandran/Images/blob/main/iris_dt_cm.png?raw=true)

#### ACCURACY
Model has achieved 93.33% accuracy from confusion matrix!

### 2. RANDOM FOREST
Verifying performance using ‘randomForest’ package.

`iris_train_class <- factor(train_data$class,`<br> 
                     `levels = c ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), `<br>
                     `labels = c (1, 2, 3)) `<br>
`iris_test_class <- factor(test_data$class,`<br> 
                     `levels = c ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), `<br>
                     `labels = c (1, 2, 3))`<br> 
 
`iris_random <- randomForest(iris_train_class ~ .,  data = train_data)`<br>
`print(iris_random)`<br>
`print (importance(iris_random,type = 2))`<br>

![random forest](https://github.com/Athira-M-Chandran/Images/blob/main/iris_random1.png?raw=true)

GINI is a measure of node impurity. From the above details it is clear that Petal features are 
more important compared to sepal features since the values are too small for sepal features 
(4.24 and 0.38) and the error rate is 1.11%. So, we can eliminate sepal feature and check the 
accuracy again.

`iris_random1 <- randomForest(iris_train_class ~ petal.length + petal.width, data = train_data )`<br>
`print(iris_random1)`<br>

![iris_random1](https://github.com/Athira-M-Chandran/Images/blob/main/iris_random2.png?raw=true)

In above table the error rate is 4.44%.

Checking the accuracy using a confusion matrix by comparing predictions to actual 
classifications. ‘caret’ package is used for confusion matrix.

`iris_pred_rand <- predict(object = iris_random1, newdata = test_data, type = "class")` <br>
`confusionMatrix(data = as.factor(iris_pred_rand), reference = as.factor(iris_test_class))`

![Confusion matrix of random forest](https://github.com/Athira-M-Chandran/Images/blob/main/iris_rnd_cm.png?raw=true)

#### ACCURACY

In above result, the accuracy is 0.9333  
So, the accuracy for this model is (0.9333 * 100)% =93.33% 

### 3. NAIVE BAYES MODEL
Naive Bayes is a classification technique based on Bayes’ Theorem with an assumption of 
independence among predictors. For Naïve Bayes model , ‘e1071’ package is used. 

`classifier_cl <- naiveBayes(class ~ ., data = train_data)`<br>
`y_pred <- predict(classifier_cl, newdata = test_data) #predicting on test data` <br>
`cm <- table(test_data$class, y_pred) #for confusion matrix` <br>
`confusionMatrix(cm) #model evaluation` <br>
![confusion matrix of naive bayes](https://github.com/Athira-M-Chandran/Images/blob/main/iris_naive_cm.png?raw=true)

#### ACCURACY
In above result the accuracy is 0.95
Accuracy of model is (0.95 * 100 )= 95%

### 4. LOGISTIC REGRESSION
We are taking first two classes from dataset, I.e, iris setosa and iris versicolor since the response 
variable in logistic regression should be categorical values.

`iris_dataset <- iris[1:100,]`

Iris setosa is labelled as 0 and iris versicolor is labelled as 1.

`iris_dataset_class <- factor(iris_dataset$class,`<br>
                             `levels = c ('Iris-setosa', 'Iris-versicolor'), `<br>
                             `labels = c (0,1)) `<br>

Computing logistic regression using Generalized linear model function.

`iris_log <- glm(iris_dataset_class ~ . , family = binomial(link = "logit"), data = iris_dataset)`<br>
`summary(iris_log)    # Output`  

Checking the accuracy using a confusion matrix by comparing predictions to actual 
classifications. ‘caret’ package is used for confusion matrix.

`iris_pred_log <- predict(object = iris_log, newdata = iris_dataset, type = "response")`

Since the values vary from 0 to 1, values above 0.5 are taken as 1 and values below 0.5 are 
taken as 0.

`iris_glm <- ifelse(iris_pred_log > 0.5 , 1, 0)`

Checking the accuracy using confusion matrix

`confusionMatrix(as.factor(iris_glm), as.factor(iris_dataset_class))`
![confusion matrix of logistic regression](https://github.com/Athira-M-Chandran/Images/blob/main/iris_log_cm.png?raw=true)

#### ACCURACY
In the above result accuracy is 1 <br>
i.e., our model has achieved 100% accuracy!

### INFERENCE
Accuracy: <br>
 Decision tree - 93.33% <br>
 Random Forest - 93.33% <br>
 Naive Bayes   - 95% <br>
 Logistic Regression - 100%<br>
From above result it is evident that Logistic Regression is more accurate!
