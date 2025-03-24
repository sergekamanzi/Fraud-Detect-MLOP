import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from preprocessing import load_data, preprocess_data

def build_model(input_shape):
    """Build and compile the neural network model."""
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, X_val, Y_val, class_weight_dict):
    """Train the model with early stopping."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=32,
                        class_weight=class_weight_dict, callbacks=[early_stopping], verbose=1)
    return history

def save_model_and_scaler(model, scaler, model_path='fraud_detection_model.keras', scaler_path='standard_scaler.pkl'):
    """Save the trained model and scaler."""
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}, Scaler saved to {scaler_path}")

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('../data/train/fraud detection.csv')
    X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, class_weight_dict, numeric_cols = preprocess_data(data)
    
    # Build and train model
    model = build_model(X_train.shape[1])
    history = train_model(model, X_train, Y_train, X_val, Y_val, class_weight_dict)
    
    # Save model and scaler
    save_model_and_scaler(model, scaler)