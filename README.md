# Cancer-cell-detection
Breast Cancer Classification using Decision Trees

This project focuses on classifying breast cancer cases as either benign or malignant using machine learning techniques, specifically a Decision Tree Classifier. The dataset used is the Breast Cancer dataset from the scikit-learn library.

Introduction

This project utilizes a Decision Tree model to classify breast cancer instances as either benign or malignant. The project explores different machine learning techniques, including splitting the dataset into training and testing sets, model training, prediction, and evaluation using metrics like accuracy and confusion matrix.

Technologies Used

Python
Scikit-learn
Matplotlib
Numpy
Jupyter Notebook
Project Structure

The project consists of the following key steps:

Loading the Breast Cancer dataset from scikit-learn.
Preprocessing the data for training.
Splitting the data into training and testing sets.
Training a Decision Tree Classifier.
Evaluating the model's accuracy and generating a confusion matrix.
Visualizing the decision tree and confusion matrix.
Dataset

The Breast Cancer dataset from scikit-learn contains 30 features computed from a digitized image of a breast mass. The features describe characteristics such as radius, texture, and smoothness. The target labels represent two classes:

0: malignant
1: benign
Model Training and Testing

Train-Test Split: The dataset is split into training (67%) and testing (33%) sets using train_test_split.
Decision Tree Classifier: The model is trained using the DecisionTreeClassifier with the "entropy" criterion.
Accuracy: After training, the model's accuracy on the test set is computed to be 95.21%.
Confusion Matrix: The confusion matrix is plotted to provide a detailed view of the model's performance.
Results

Accuracy: The model achieved an accuracy of 95.21%.
Confusion Matrix: The confusion matrix provides insights into the model's predictions and misclassifications.
Confusion Matrix Example

Decision Tree Visualization: The decision tree is plotted for better understanding of the model's decision process.
Decision Tree Example

Usage

Open the Jupyter notebook and run the cells to load the dataset, train the model, and visualize results.

jupyter notebook breast_cancer_classification.ipynb
To visualize the decision tree and confusion matrix, run the respective cells in the notebook.

Conclusion

This project successfully demonstrates the use of a Decision Tree model in classifying breast cancer data with high accuracy. Future improvements could involve experimenting with other machine learning models and feature engineering techniques to further enhance performance
