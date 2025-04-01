import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/content/fraud detection.csv')

# Drop unnecessary columns
data = data.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])

# Encode 'type' column
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# Scale numeric columns
scaler = StandardScaler()
numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Split features and target
X = data.drop(columns=['isFraud'])
Y = data['isFraud']

# Split into train, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Save datasets
X_train.to_csv('X_train.csv', index=False)
Y_train.to_csv('Y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
Y_test.to_csv('Y_test.csv', index=False)

# Save scaler
joblib.dump(scaler, 'standard_scaler.pkl')