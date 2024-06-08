---
layout: project
type: project
image: img/projects/BC_AoU/aou_logo.png
title: Machine Learning Approaches to Breast Cancer Classification with All of Us Data
permalink: projects/breast-cancer-classification-aou
date: 2024-05-10
labels:
  - Python
  - Machine Learning
  - All of Us Program
  - MLP
  - SVC
  - Random Forest
  - Adaboost
  - Gradient boosting models
summary: "The All of Us program is a program organized by the (United States) National Institute of Health which serves to aggregate, anonymize, and make available patient health data for research projects. Using the All of Us database, I built several classification models to predict malignancy in breast cancer patients, as well as compare differences in model performance between two differentiated dataset groups."
---
<h2>Purpose and Overview</h2>
This project was my Honors undergraduate thesis at the University of Hawai'i at Mānoa, in which I used Python to prepare and analyze publicly available breast cancer patient data to train machine learning models to predict the malignancy of these patients. The data was sourced using the All of Us program, a program organized by the (United States) National Institute of Health which serves to aggregate, anonymize, and make available patient health data for research projects. This project can be separated into three major stages: data cleaning and preparation, exploratory data analysis (EDA), and model training and evaluation. The machine learning model types used in this study were imported in the sklearn library and include: multilayer perceptron (MLP), support vector machine classifier (SVC), random forest, Adaboost classifier, and gradient boosting classifier.

<h3>Specific Aims</h3>
This study had two main goals: the first was to successfully train a classification model to have an AUC-ROC score of 0.85, which would be indicative a well-trained model, and the second was to compare the model peformance difference between two different types of datasets, a smaller more complex dataset group vs. a relatively larger but simpler dataset group.

<h3>Dataset</h3>
  <p>For this project, I used anonymized patient data available via the [All of Us program](https://www.researchallofus.org/]. More specifically, this project uses the All of Us Registered Tier Dataset v7 availble using the All of Us cloud-computing environment. Cardiovascular health and liquid biopsy data from both bening and malingnant patients were used to construct the bulk of the training dataset. All of the predicting data was quantitative in nature and selected from the All of Us database using SQL queries. Due to the larger number of entries, data was aggregated on a per-patient/per-year basis. For example, if patient 001 has entries in the years 2000, 2001, 2003, and 2006, they would have four separate entries.</p>
  <p>Additionally, Fibit data was used to construct more complex datasets. These datasets were smaller than their original counterparts and included several additional training features to test the model performance of these smaller more complex datasets against their larger, simpler counterparts.</p>

<h3>Preparatory Analysis</h3>
Because of the dataset's small size, and partly because I wanted to experiment, I used feature expansion to test if the logit models trained on more parameters would work better. Furthermore, I used the LASSO algorithm to compute which features, among those from the original features in the dataset and those from feature expansion, would be the most statistically significant in determining diagnosis.

One thing to note is that, because of the dataset's small size, it has relatively low external validity and is prone to overfitting. For this reason, this project is used more as an exercise in data science rather than building actually usable models.




<h2>Conclusions</h2>
Overall, this project was an excellent experience in dabbling into data science projects, as well as planning and executing the data analysis by myself. However, I did not make any ROC or AUROC curves, and I did not have the technical experience to try a GBM model. In my future endeavors, I hope to perform a more thorough analysis that includes these elements, if appropriate.

<h2>Links</h2>
View the open-source [breast cancer dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download).

View the full project write-up, including source code, <a href="../documents/BC_class_in_R_report.html">here</a>.
