#!/usr/bin/env python3
"""
Simple script to save your TensorFlow model for deployment
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def save_model():
    """Train and save your TensorFlow model"""
    
    print("Loading data...")
    df = pd.read_csv("bank_additional_full_cleaned.csv")
    df = df.drop_duplicates()
    print(f"Data shape: {df.shape}")
    
    # Define features
    features = [
        'age','job','marital','education','default','housing','loan',
        'contact','month','day_of_week','campaign','pdays','previous','poutcome',
        'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'
    ]
    
    X = df[features]
    y = df['y'].map({'yes': 1, 'no': 0})
    
    # Define categorical and numerical columns
    cat_cols = ['job','marital','education','default','housing','loan',
                'contact','month','day_of_week','poutcome']
    num_cols = [col for col in X.columns if col not in cat_cols]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )
    
    # Process features
    X_processed = preprocessor.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Create and train model
    print("Training TensorFlow model...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        verbose=1,
        class_weight=class_weight_dict
    )
    
    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save model using TensorFlow's native method
    print("Saving model...")
    try:
        # Save as H5 format (more compatible)
        model.save('tensorflow_model.h5')
        print("✓ TensorFlow model saved as H5 format!")
    except Exception as e:
        print(f"Error saving TensorFlow model: {e}")
        print("Trying alternative save method...")
        # Alternative: Save weights only
        model.save_weights('tensorflow_weights.h5')
        # Save model architecture as JSON
        model_json = model.to_json()
        with open('tensorflow_model.json', 'w') as json_file:
            json_file.write(model_json)
        print("✓ Model saved as weights + JSON format!")
    
    # Save preprocessor and feature names
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(features, 'feature_names.pkl')
    
    print("✓ All files saved successfully!")
    print("Files created:")
    print("- tensorflow_model/ (directory) or tensorflow_model.h5")
    print("- preprocessor.pkl")
    print("- feature_names.pkl")

if __name__ == "__main__":
    save_model()
