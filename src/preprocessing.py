import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

def load_data(file_path):
    data = pd.read_csv("/content/fraud detection.csv")
    return data

def preprocess_data(data):
    """Preprocess the dataset based on the Colab workflow."""
    # Drop unnecessary columns
    data = data.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])
    
    # Encode 'type' column using LabelEncoder
    le = LabelEncoder()
    data['type'] = le.fit_transform(data['type'])
    
    # Define numeric columns for scaling (excluding 'isFraud')
    numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    # Initialize and apply StandardScaler to numeric columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Separate features (X) and target (Y)
    X = data.drop(columns=['isFraud'])
    Y = data['isFraud']
    
    # Split data into train, validation, and test sets (70% train, 15% val, 15% test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, class_weight_dict, numeric_cols

def explore_data(data):
    """Explore the dataset with visualizations and statistics as in Colab."""
    # Print unique values of 'type' column before encoding
    unique_values = data['type'].unique()
    print("Unique transaction types (original):", unique_values)
    
    # Basic statistics
    print("\nDataset Description:")
    print(data.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    # Encode 'type' for further exploration
    le = LabelEncoder()
    data['type'] = le.fit_transform(data['type'])
    print("\nUnique transaction types (encoded):", data['type'].unique())

     # Correlation heatmap (excluding 'isFraud')
    corr_matrix = data.drop(columns=['isFraud']).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap (Without Target Columns)')
    plt.show()
    
    # Transaction type distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='type', data=data)
    plt.title('Transaction Type Distribution')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.show()
    
    # Boxplot of amount by fraud status
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='isFraud', y='amount', data=data)
    plt.title('Transaction Amount Distribution by Fraud/Non-Fraud')
    plt.xlabel('Is Fraud')
    plt.ylabel('Transaction Amount')
    plt.show()
    
    # Dataset shape
    print("\nDataset Shape:", data.shape)
