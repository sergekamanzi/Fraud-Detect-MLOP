# Fraud-Detect-MLOP
video = 

website link = https://fraudent-web-page.vercel.app/

api swagger UI= https://fraud-backend-283k.onrender.com/docs

full folder API = https://github.com/sergekamanzi/fraud-backend.git

full folder Frontend = https://github.com/sergekamanzi/FraudentWebPage.git


# Fraud Detection Using Neural Networks

## Project Overview
This project implements a fraud detection system using a neural network built with TensorFlow and Keras. The goal is to classify transactions as fraudulent or non-fraudulent based on features like transaction type, amount, and account balances. The dataset is preprocessed, visualized for insights, and used to train a model with techniques to handle class imbalance.

## Dataset
The dataset (`fraud_detection.csv`) contains 7050 transactions with the following columns:
- `step`: Time step of the transaction
- `type`: Transaction type (CASH_OUT, TRANSFER, CASH_IN, PAYMENT, DEBIT)
- `amount`: Transaction amount
- `nameOrig`: Originating account identifier
- `oldbalanceOrg`: Originating account balance before transaction
- `newbalanceOrig`: Originating account balance after transaction
- `nameDest`: Destination account identifier
- `oldbalanceDest`: Destination account balance before transaction
- `newbalanceDest`: Destination account balance after transaction
- `isFraud`: Target variable (1 for fraudulent, 0 for non-fraudulent)
- `isFlaggedFraud`: Flag for transactions suspected as fraudulent

### Dataset Insights
- **Class Imbalance**: Approximately 29% of transactions are fraudulent (2050), and 71% are non-fraudulent (5000).
- **Transaction Types**: CASH_OUT transactions are the most common, while DEBIT transactions are rare.
- **Transaction Amounts**: Non-fraudulent transactions show more outliers with higher amounts compared to fraudulent ones.

## Model
- **Architecture**: A Sequential neural network with Dense layers, L2 regularization, and Adam optimizer.
- **Class Imbalance Handling**: Class weights are computed using `compute_class_weight` to balance the training process.
- **Training**: The model is trained with early stopping to prevent overfitting.

## Results
- The model is evaluated using accuracy, precision, recall, F1-score, and confusion matrix on the test set.
- Results will be printed after running the script.

