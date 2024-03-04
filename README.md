# Module 12 Report Template

## Overview of the Analysis

The analysis aims to build a predictive model for credit risk classification, predicting whether a loan is healthy or high-risk based on various financial attributes of borrowers.

The dataset includes financial attributes such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt, etc. The prediction target is the loan_status variable, indicating whether a loan is healthy (0) or high-risk (1).

Key observations:

The loan_status variable is balanced, with an equal number of healthy (0) and high-risk (1) loans.
Logistic Regression is used as one of the classification algorithms.
Resampling methods like Random Oversampling may have been employed to handle class imbalance, as indicated by balanced accuracy and confusion matrices with balanced classes.

Stages of the Machine Learning Process:

- Data Preparation: The analysis began with the importation of necessary modules such as numpy, pandas, and pathlib. These modules facilitate data manipulation and file handling, crucial steps in preparing the dataset for analysis. Additionally, the train_test_split function from sklearn.model_selection was utilized to split the dataset into separate training and testing subsets.

- Model Evaluation: Metrics for model evaluation were imported from scikit-learn, including 'balanced_accuracy_score', 'confusion_matrix', and 'classification_report'. These metrics are essential for assessing the performance of machine learning models.

Methods Used:

- Logistic Regression: The analysis utilized logistic regression, an algorithm suitable for binary classification tasks like credit risk classification. Logistic regression was provided and fitted to the training data, and its predictions were evaluated against the test data.

- Resampling Method: Due to the imbalanced nature of the dataset, a resampling technique was employed to address this issue. Specifically, the RandomOverSampler method from the imbalanced-learn library was used to create a resampled training dataset. This technique helps mitigate the impact of class imbalance on model performance, leading to more accurate predictions.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

- Accuracy:
Model 1 achieves an accuracy of 99%.
This means that 99% of the predictions made by Model 1 are correct.

- Precision:
For class 0 (healthy loan), the precision is 100%.
This indicates that when Model 1 predicts a loan as healthy, it is correct 100% of the time.
For class 1 (high-risk loan), the precision is 88%.
This means that when Model 1 predicts a loan as high-risk, it is correct 88% of the time.

- Recall:
For class 0 (healthy loan), the recall is 100%.
This indicates that Model 1 correctly identifies all actual healthy loans.
For class 1 (high-risk loan), the recall is 92%.
This means that Model 1 correctly identifies 92% of the actual high-risk loans.


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
- Accurancy: 
Model 2 achieves an accuracy of 100%.
This means that all of the predictions made by Model 2 are correct.

- Precision:
For class 0 (healthy loan), the precision is 100%.
This indicates that when Model 2 predicts a loan as healthy, it is correct 100% of the time.
For class 1 (high-risk loan), the precision is 88%.
This means that when Model 2 predicts a loan as high-risk, it is correct 88% of the time.

- Recall:
For class 0 (healthy loan), the recall is 100%.
This indicates that Model 2 correctly identifies all actual healthy loans.
For class 1 (high-risk loan), the recall is 99%.
This means that Model 2 correctly identifies 99% of the actual high-risk loans.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? ) If you do not recommend any of the models, please justify your reasoning.

In summary, the choice of which model performs best depends on the specific objectives of the analysis and the relative importance of correctly predicting each class. Model 2, with its perfect accuracy and precision for predicting healthy loans, might be preferred in scenarios where overall correctness and balanced performance across classes are crucial. However, in scenarios where minimizing false negatives for high-risk loans is the primary concern, Model 1, with its higher recall for class 1, might be preferred.


