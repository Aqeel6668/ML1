# Spam SMS Detection

This project uses a Naive Bayes classifier to detect spam messages from text.

## Dataset
Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Features
- Message text (cleaned and vectorized)
- Labels: 'spam' or 'ham'

## Workflow
1. Clean text (lowercase, remove punctuation)
2. Convert text to vectors using TF-IDF
3. Train a Multinomial Naive Bayes model
4. Evaluate using classification report

## Tools
- Python
- scikit-learn
- pandas
