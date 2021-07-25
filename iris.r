#PACKAGES

# Visualization
library(ggplot2)
library(gridExtra)
library(grid)

# splitting of dataset
library(caTools)

#Random Forest
library(grid)
library(party)
library(randomForest)

# Confusion matrix
library(caret)
library(lattice)
library(partykit)

#Decision tree
library(rpart)
library(rpart.plot)

#Naive bayes
library(klaR) 
library(e1071)

#Loading dataset

iris<- read.csv("c:/data/iris.csv")

#For Viewing dataset

View(iris)   
str(iris)     #view structure of dataset
summary(iris) #view statistical summary of dataset
head(iris)    #view top 6 rows of dataset

#######################################################################
                      # DATA VISUALIZING #
#######################################################################

# Analysis with Histogram,

# Sepal length 
sepal_len <- ggplot(data=iris, aes(x=sepal.length))+
  geom_histogram(binwidth=0.2, color="black", aes(fill=class)) + 
  xlab("Sepal Length (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="none")

# Sepal width
sepal_width <- ggplot(data=iris, aes(x=sepal.width)) +
  geom_histogram(binwidth=0.2, color="black", aes(fill=class)) + 
  xlab("Sepal Width (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="none")

# Petal length
petal_len <- ggplot(data=iris, aes(x=petal.length))+
  geom_histogram(binwidth=0.2, color="black", aes(fill=class)) + 
  xlab("Petal Length (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="none")

# Petal width
petal_width <- ggplot(data=iris, aes(x=petal.width))+
  geom_histogram(binwidth=0.2, color="black", aes(fill=class)) + 
  xlab("Petal Width (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="right" )

# Plot all visualizations
grid.arrange(sepal_len ,
             sepal_width ,
             petal_len,
             petal_width,
             nrow = 2,
             top = textGrob("Iris Frequency Histogram", 
                            gp=gpar(fontsize=15))
)


# set seed to ensure you always have same random numbers generated
set.seed(123)

# splits the data in the ratio mentioned in SplitRatio.
sample_data = sample.split(iris, SplitRatio = 0.75)

#creates a training dataset named train_data with rows which are marked as TRUE
train_data <- subset(iris, sample_data == TRUE) 
test_data <- subset(iris, sample_data == FALSE)

########################################################################
                       #  DECISION TREE #
#######################################################################

fit <- rpart(class ~ ., method = "class",
            data = train_data, 
            control = rpart.control(cp = 0),
            parms = list(split="information"))

#plot using rpart
rpart.plot(fit,type= 4 , extra=1)

#Checking the accuracy using a confusion matrix by comparing predictions to actual classifications

iris_pred <- predict(object = fit,
                     newdata = test_data,
                     type = "class")

confusionMatrix(data = as.factor(iris_pred),
                reference = as.factor(test_data$class))

# Table view of above
table(as.factor(test_data$class), as.factor(iris_pred) )

########################################################################
                      #  RANDOM FOREST #
########################################################################

#character to numeric
iris_train_class <- factor(train_data$class, 
                     levels = c ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), 
                     labels = c (1, 2, 3)) 
iris_test_class <- factor(test_data$class, 
                     levels = c ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), 
                     labels = c (1, 2, 3)) 

#Appyling randomForest function
iris_random <- randomForest(iris_train_class ~ ., data = train_data )
print(iris_random)
print (importance(iris_random,type = 2))

#Shows the accuracy
iris_random1 <- randomForest(iris_train_class~ petal.length + petal.width, data = train_data )
print(iris_random1)
print(importance(iris_random1,type = 2))

#predicting
iris_pred_rand <- predict(object = iris_random1,
                     newdata = test_data,
                     type = "class")

#for checking accuracy
confusionMatrix(data = as.factor(iris_pred_rand),
                reference = as.factor(iris_test_class))

#############################################################################
                      # Fitting Naive Bayes Model #
#############################################################################

classifier_cl <- naiveBayes(class ~ ., data = train_data)
classifier_cl

# Predicting on test data'
y_pred <- predict(classifier_cl, newdata = test_data)

# Confusion Matrix
table(test_data$class, y_pred)

# Model Evaluation
confusionMatrix(data = as.factor(y_pred),
                reference = as.factor(test_data$class))

#####################################################################
# LOGISTIC REGRESSION #
######################################################################

# Taking two classes (Iris-setosa and Iris-versicolor)
iris_dataset <- iris[1:100,]

# Labeling iris-setosa as 0  and Iris-versicolor as 1
iris_dataset_class <- factor(iris_dataset$class,
                             levels = c ('Iris-setosa', 'Iris-versicolor'), 
                             labels = c (0,1)) 

# Computing logistic regression using Generalized linear model function
iris_log <- glm(iris_dataset_class ~ . , family = binomial(link = "logit"), data = iris_dataset)
# Output
summary(iris_log)

# Prediction
iris_pred_log <- predict(object = iris_log, newdata = iris_dataset, type = "response")

iris_glm <- ifelse(iris_pred_log > 0.5 , 1, 0)

# Checking the accuracy using confusion matrix
confusionMatrix(as.factor(iris_glm), as.factor(iris_dataset_class))
