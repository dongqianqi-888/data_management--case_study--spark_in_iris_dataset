# data_management--case_study--spark_in_iris_dataset
*logistic regression, decision tree and random forest in iris dataset using Spark.*

---

## 1. Introduction
This report presents a classification model of the iris dataset based on Spark statements on the Colab online platform. The process includes data cleaning, model selection, model evaluation, and prediction.

---

## 2. Dataset and methodnology
The dataset used in this study is the classic iris dataset in machine learning, sourced from the UCI public database. The feature vectors include sepal length, sepal width, petal length, petal width, and species. The 150 observations cover three flower categories: Setosa, Versicolor, and Virginica. The following are the first five rows of the dataset and the data types of each feature.
```text
---- iris ----
+------------+-----------+------------+-----------+-----------+
|sepal_length|sepal_width|petal_length|petal_width|    species|
+------------+-----------+------------+-----------+-----------+
|         5.1|        3.5|         1.4|        0.2|Iris-setosa|
|         4.9|        3.0|         1.4|        0.2|Iris-setosa|
|         4.7|        3.2|         1.3|        0.2|Iris-setosa|
|         4.6|        3.1|         1.5|        0.2|Iris-setosa|
|         5.0|        3.6|         1.4|        0.2|Iris-setosa|
+------------+-----------+------------+-----------+-----------+
only showing top 5 rows

root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 ```
For subsequent modeling and analysis, I first performed some simple cleaning and consolidation of the dataset. Since no true values ​​or obvious outliers were detected, only a minor modification was made to the data representation. Species, as a categorical variable, was re-encoded, with setosa corresponding to the number 0, versicolor to the number 1, and virginica to the number 2. 

Furthermore, because machine learning models inherently lack interpretability, for ease of modeling, I packaged all numerical variables into a single vector, features. The first five rows of the cleaned dataset are shown below.
```text
+-----------------+-----+
|         features|label|
+-----------------+-----+
|[5.1,3.5,1.4,0.2]|  0.0|
|[4.9,3.0,1.4,0.2]|  0.0|
|[4.7,3.2,1.3,0.2]|  0.0|
|[4.6,3.1,1.5,0.2]|  0.0|
|[5.0,3.6,1.4,0.2]|  0.0|
+-----------------+-----+
only showing top 5 rows
```
To prevent overfitting, I divided the 150 data points into a training set and a test set. The training set comprised 70% of the total data, while the test set consisted of the remaining 46 observations.

To identify the flower variety for each sample size, we can determine it by examining the characteristics of its petals and sepals. Therefore, the features in the data serve as the model's feature variables, while the variety is used as the target variable. The classification models we use are logistic regression, decision trees, and random forests. I will now describe these methods in detail.

Logistic regression wraps an exponential function around a regular linear regression function, making the mapping of independent variables binary. During modeling, a threshold is manually defined; samples with dependent variables below this threshold belong to one class, and samples with dependent variables above this threshold belong to another. When dealing with complex classification problems, running this logic multiple times can resolve issues involving multiple classification categories. For example, first separate flowers into class A and those that are not class A; then, from the flowers that are not class A, further distinguish between flowers that are class B and those that are not class B, and so on.

Decision trees generate classification nodes based on feature variables according to the degree of decrease in the Gini coefficient, separating samples layer by layer. Here, the Gini coefficient is a metric for measuring the purity of a dataset; the smaller the Gini coefficient, the more homogeneous the categories. For example, we might first divide all samples into two classes based on the size of the petals, then continue classifying them based on the size of the sepals, and so on. This process will eventually separate the samples according to their feature categories.

Random forests are an upgraded version of decision trees. The principle is to randomly sample from the dataset and randomly select feature vectors for classification, generating a large number of independent decision tree models. Each sample is then classified by these decision trees, and the classification result depends on the majority of the decision tree judgments. For example, if more than half of the decision tree models classify the first flower as category A, then the random forest will predict that the first flower is category A.

---

## 3. Model analysis
Now that we have a strong model foundation, we use cross-validation to train the best classification mechanism for this data. Cross-validation involves repeatedly generating models using different training and test sets, and finally judging the quality of the models based on the confusion matrix to select the ultimate model. All cross-validation operations are based on the test dataset we previously split. And my cross-validation here involves splitting the data into three parts.

The confusion matrix reflects the differences between the true situation and the model's predictions, showing the number of samples where the model correctly judged and incorrectly judged. The most important metrics include precision, recall, accuracy, and the F1 score.

Accuracy refers to the percentage of samples correctly predicted by the model; precision is the probability that all predicted positive samples are correct; and recall is the probability that true positive samples are found. All three metrics are generally better the higher they are. However, higher precision does not necessarily mean higher recall. Therefore, we introduce the F1 index (the harmonic mean of precision and recall) to comprehensively evaluate model performance. The F1 index ranges from 0 to 1, with values ​​closer to 1 indicating a better model.

After understanding the principles of model selection and evaluation, we used the classification model program in PySpark machine learning to find the best model in each methods and evaluated its performance on the training set. The results are shown below:
```text
---- best model in traning set ----
{'model': 'logistic regression', 'accuracy': 0.9808, 'precision': 0.9808, 'recall': 0.9808, 'F1': 0.9808}
{'model': 'decision tree', 'accuracy': 0.9904, 'precision': 0.9906, 'recall': 0.9904, 'F1': 0.9904}
{'model': 'random forest', 'accuracy': 0.9712, 'precision': 0.9714, 'recall': 0.9712, 'F1': 0.9712}

 best one:decision tree
```
Based on the evaluation criteria, the decision tree model outperforms all other models, making it the most ideal among the three based on the training set results. However, to avoid overfitting—where a model heavily relies on the training data and performs poorly on the test set—we re-evaluate these three optimal models using the test set.
```text
---- model performance on test set ----
{'model': 'logistic regression', 'accuracy': 0.9348, 'precision': 0.9364, 'recall': 0.9348, 'F1': 0.9349}
{'model': 'decision tree', 'accuracy': 0.913, 'precision': 0.9335, 'recall': 0.913, 'F1': 0.9126}
{'model': 'random forest', 'accuracy': 0.9565, 'precision': 0.9623, 'recall': 0.9565, 'F1': 0.9566}

 best one on test set: random forest
```
As can be seen, the performance of logistic regression and decision trees on the test set has declined significantly compared to before, while the random forest model, which scored the worst on the training set, shows strong stability.
A preview of the detailed prediction results is displayed in Colab.

To train a better model, I reviewed the entire training process. The old model's parameters could cover most situations, but the number of cross-validations was still too low. To explore whether increasing the number of cross-validations could improve the model's accuracy, I increased the number to 10. I then searched for the optimal model without changing the remaining parameters. The following shows the performance of the new model on the training and test sets.
```text
---- Best Model in Training Set ----
{'model': 'Logistic Regression', 'accuracy': 0.9808, 'precision': 0.9808, 'recall': 0.9808, 'F1': 0.9808}
{'model': 'Decision Tree', 'accuracy': 0.9904, 'precision': 0.9906, 'recall': 0.9904, 'F1': 0.9904}
{'model': 'Random Forest', 'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'F1': 1.0}

Best model on training set: Random Forest

---- Model Performance on Test Set ----
{'model': 'Logistic Regression', 'accuracy': 0.9348, 'precision': 0.9364, 'recall': 0.9348, 'F1': 0.9349}
{'model': 'Decision Tree', 'accuracy': 0.913, 'precision': 0.9335, 'recall': 0.913, 'F1': 0.9126}
{'model': 'Random Forest', 'accuracy': 0.9348, 'precision': 0.947, 'recall': 0.9348, 'F1': 0.9348}

Best model on test set: Logistic Regression
```
We can see that among the three models, only the best Random Forest model changed, achieving an astonishing 100% accuracy on the test set, although its score on the test set decreased. Logistic Regression ultimately surpassed the Random Forest model with a 0.0001 F1 score advantage, winning the top spot.
```test
Time taken when the numFolds of cross-validations is 3:
============================================================
              MODEL TRAINING TIME REPORT              
============================================================
Logistic Regression Training Time: 25.73 sec
Decision Tree Training Time:       29.93 sec
Random Forest Training Time:       23.85 sec
Total Running Time:                83.17 sec
============================================================

Time taken when the numFolds of cross-validations is 10:
============================================================
              MODEL TRAINING TIME REPORT              
============================================================
Logistic Regression Training Time: 74.23 sec
Decision Tree Training Time:       64.38 sec
Random Forest Training Time:       76.53 sec
Total Running Time:                217.54 sec
============================================================
```
I used the `time` package to calculate the time taken for the two cross-validation methods, as shown above. It can be seen that cross-validation divided into 10 parts takes almost three times longer than cross-validation divided into 3 parts. Therefore, it can be inferred that `numFolds` and the time taken have a linear relationship.

---

## 4. Conclusion
This article details my process of using Spark and three machine learning models to find the optimal classification model, and briefly explains the concepts involved. The main conclusions are as follows:

- When comparing the results of logistic regression, decision trees, and random forests, the performance ranking on the training set may be completely opposite to the performance on the dataset.
- When comparing the results of logistic regression, decision trees, and random forests, the performance ranking on the training set may be completely opposite to the performance on the dataset.
- When solving the iris classification problem, increasing the number of folds for cross-validation from 3 to 10 did not improve the performance of the three models used.
- The number of folds for cross-validation may have a linear relationship with the model training time.
