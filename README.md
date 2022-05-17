# Credit_Risk_Analysis
Module 17 Challenge

## Overview

In this module we use machine learning tools <scikit-learn> and <imbalanced-learn> imported into a python environment using <Anaconda> to train and evaluate models that estimate credit card risk. The credit card dataset is from LendingClub, which is a peer-to-peer lending services company. 
  
In this module, we used 6 techniques to train and evaluate models with unbalanced classes to determine the overall credit risk: 
* Oversampling with the RandomOverSampler algorithm
* Oversampling with the SMOTE algorithm
* Undersampling with the ClusterCentroids algorithm
* Over and Under sampling with the SMOTEENN algorithm
* BalancedRandomForestClassifer machine learning model to reduce bias
* EasyEnsembleClassifier machine learning model to reduce bias
  
## Results
  
### RandomOverSampler algorithm
  
RandomOverSampler was used for the Naive Random Oversampling method to resample data and enlarge the smaller class of training data to use in a logistic regression model. This regression model was trained and the image below displays the resulting balanced accuracy score, confusion matrix, and classification report. 
  
  <img width="896" alt="Naive Random Oversampling" src="https://user-images.githubusercontent.com/96043107/168726516-2ae5ff14-5f5f-4b8b-acce-3185ab2677cf.png">

* With a balanced accuracy score of 64.9%, the model therefore predicted the credit risk score accurately 64.9% of the time. This is relatively positive, but not great. 
* The precision scores indicate that they are skewed towards the low-risk loans due to all of the low-risk loans being correctly predicted. However, none of the high-risk loans were correctly predicted, which means that this model is poor at identifying high-risk loans. 
  
### SMOTE algorithm
  
The SMOTE algorithm was used for the SMOTE Oversampling method to resample the data and enlarge the smaller class of training data to use in a logistic regression model. This regression model was trained and the image below displays the resulting balanced accuracy score, confusion matrix, and classification report. 
  
 <img width="885" alt="SMOTE Oversampling" src="https://user-images.githubusercontent.com/96043107/168727365-b00f42a9-e7a7-4d14-9a92-256b94aeb33e.png">

* With a balanced accuracy score of 64.8%, the model therefore predicted the credit risk score accurately 64.8% of the time. This is relatively positive, but not great. 
* The precision scores indicate that they are skewed towards the low-risk loans due to all of the low-risk loans being correctly predicted. However, none of the high-risk loans were correctly predicted, which means that this model is poor at identifying high-risk loans. 
* The recall for both low and high risk loans were about 63%, meaning that the model found a positive instance 63% of the time. This isn't a great score when attempting to identify high-risk loans. 
  
### ClusterCentroids algorithm
  
The ClusterCentroids algorith was used for the undersampling method in order to resample the data and reduce the larger class of training data to use in a logistic regression model. This regression model was trained and the image below displays the resulting balanced accuracy score, confusion matrix, and classification report. 
  
<img width="743" alt="Undersampling" src="https://user-images.githubusercontent.com/96043107/168728051-152c163e-49a6-4db1-9b14-c8dc83374fa3.png">

* With a balanced accuracy score of about 61.2%, which means that 61.2% of the testing data was correctly classified. This is not a great  accuracy score. 
* Precision scores for the model are skewed towards the low-risk loans, and all of the low-risk loans were correctly predicted. However, almost all of the high-risk loans were incorrectly predicted, and therefore the model is poor in identifying high-risk loans. 
* The high-risk loan recall score was 0.59, and the recall scores for low-risk loans was 0.44, meaning that while the model is better at precicting high-risk loans, both of these scores are low and prove that the model does not accurately predict the positives about half the time on average between high-risk and low-risk loans. 
  
### SMOTEENN algorithm

The SMOTEENN algorithm was used for the Combination method, using both over and under sampling. This algorithm removes outliers and resamples the data for a logistic regression model. The logistic regression model was trained and the following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

<img width="815" alt="Combination (Over   under) Sampling" src="https://user-images.githubusercontent.com/96043107/168729200-08c65e14-fd3d-4686-9329-df7af067685d.png">
  
* With a balanced accuracy score of about 51.1%, we can say that only 51.1% of the testing data was accurately classified. This is a lower accuracy score than the above techniques. 
* The prediction scores for this model are again skewed towards low-risk loans, which were correctly predicted. Again, however, high-risk loans were rarely accurately predicted in this model. 
* The recall scores for the high-risk and low-risk loans were 0.70 and 0.58, respectively, which compared to previous models is much better at predicting both high and low risk loans. 
  
### BalancedRandomForestClassifer 

The Balanced Random Forest Classifier model was used to create 100 decision trees that classify the testing data. The following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report. 
  
<img width="817" alt="Balanced Random Forest Classifier" src="https://user-images.githubusercontent.com/96043107/168729767-06e53c6f-a6bd-47d2-b5d5-a3062c294d35.png">

* With a balanced accuracy score of about 78.8%, we can say that 78.8% of the testing credit data was accurately classified. Compared to previous above models, this score is much higher than others we have discussed. 
* The prediction scores are again skewed towards low-risk loans, which were all correctly predicted. Meanwhile, this model stil does not correctly predict high-risk loans with any frequency. 
* The recall score for high-risk loans is 0.67, and the recall score for low-risk loans is 0.91, which are both fairly high scores, the low-risk loans in particular. This model is good at predicting true positives for low-risk loans. 
  
### EasyEnsembleClassifier 
The Easy Ensemble Classifier was used to train and evaluate models to classify the testing data. The following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

<img width="848" alt="Easy Ensemble AdaBoost Classifier" src="https://user-images.githubusercontent.com/96043107/168730282-45bc724d-10ab-4f77-a9c6-1d871329b342.png">

* With a balanced accuracy score of about 92.5%, we can say that the classifier accurately predicted the credit risk 92.5% of the time. This score is higher than in any other model we've used. 
* The precision scores for the model are again skewed towards low-risk loans, which were predicted correctly, but again the high-risk loans were still mainly predicted incorrectly. 
* The recall scores for the high-risk loans was 0.91, and for the low-risk loans 0.94, meaning that the model has a high rate of true positives for both categories. 

## Summary 
This analysis shows that the ensemble model is better that the resampling models in all three categories: balanced accuracy score, recall scores, and precision scores. 
* Worst Performing model: SMOTEENN algorithm
* Best Performing model: Easy Ensemble Classifier
 
Even with a higher level of success in the ensemble model, the precision scores were still poor. This means that the model would still give too many false positives if it were to be used in the real-world. If you are in the business of lending money, the repercussions of a poor precision score could be very large, and its not great if the model is turning away potential clients with good credit risks. 

