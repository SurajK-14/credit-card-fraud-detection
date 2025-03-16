# Credit Card Fraud Detection

## Project Overview
This project implements and compares various machine learning models to detect fraudulent credit card transactions. The models include Logistic Regression, Neural Networks, Random Forest, Gradient Boosting, and Support Vector Machines.

## Data Description

### Data Source
The dataset used is the Credit Card Fraud Detection dataset from Kaggle. It contains transactions made by European cardholders in September 2013.
Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Data Preparation
- Loaded the dataset using Kaggle API
- Applied RobustScaler to normalize the 'Amount' field
- Normalized the 'Time' feature between 0 and 1
- Shuffled the data randomly for better training
- Split the data into training (85%), testing (8%), and validation (7%) sets
- Created a balanced dataset through downsampling to address class imbalance

## Methodology

### Exploratory Data Analysis
Initial data visualization and statistical analysis

### Data Preprocessing
Normalization and train-test-validation split

### Model Implementation
- Logistic Regression
- Shallow Neural Networks
- Random Forest
- Gradient Boosting
- Support Vector Machines

### Model Evaluation
Comparison using precision, recall and F1-score with emphasis on fraud detection

## Installation & Setup

```bash
# Install required packages
pip install kaggle pandas numpy matplotlib seaborn scikit-learn tensorflow

# Configure Kaggle API
# (Requires kaggle.json file in the project directory)
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download and extract the dataset
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
```

## Results
The models performed differently on balanced and imbalanced datasets:

- Logistic Regression achieved high precision (0.83) on the imbalanced dataset
- Neural Networks showed better recall (0.78) on the imbalanced dataset
- After downsampling, the models generally showed more balanced precision and recall scores
- Gradient Boosting performed well on both imbalanced and balanced datasets
