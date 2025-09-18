#!/usr/bin/env python3
"""
Fix TensorFlow model for deployment - ANN only
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_ann_model():
    """Create ANN model that will work on Render"""
    
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
    
    # Create ANN model (compatible syntax)
    print("Training ANN model...")
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
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
    
    # Save model using H5 format (most compatible)
    print("Saving ANN model...")
    try:
        # Save as H5 format
        model.save('ann_model.h5')
        print("✓ ANN model saved as H5 format!")
    except Exception as e:
        print(f"Error with H5: {e}")
        # Fallback: Save weights and architecture separately
        model.save_weights('ann_weights.h5')
        model_json = model.to_json()
        with open('ann_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        print("✓ ANN model saved as weights + architecture!")
    
    # Save preprocessor and feature names
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(features, 'feature_names.pkl')
    
    print("✓ All files saved successfully!")
    print("Files created:")
    print("- ann_model.h5")
    print("- preprocessor.pkl")
    print("- feature_names.pkl")
    
    # Test the saved model
    print("Testing saved model...")
    try:
        from tensorflow.keras.models import load_model
        loaded_model = load_model('ann_model.h5')
        loaded_preprocessor = joblib.load('preprocessor.pkl')
        loaded_features = joblib.load('feature_names.pkl')
        
        # Test with sample data
        test_data = pd.DataFrame([{
            'age': 35,
            'job': 'admin.',
            'marital': 'married',
            'education': 'university.degree',
            'default': 'no',
            'housing': 'yes',
            'loan': 'no',
            'contact': 'cellular',
            'month': 'may',
            'day_of_week': 'thu',
            'campaign': 1,
            'pdays': 999,
            'previous': 0,
            'poutcome': 'nonexistent',
            'emp.var.rate': 1.1,
            'cons.price.idx': 93.2,
            'cons.conf.idx': -36.4,
            'euribor3m': 4.9,
            'nr.employed': 5228.1
        }])
        
        # Ensure all features are present
        for feature in loaded_features:
            if feature not in test_data.columns:
                test_data[feature] = 0
        
        test_data = test_data[loaded_features]
        X_test_processed = loaded_preprocessor.transform(test_data)
        prediction = loaded_model.predict(X_test_processed, verbose=0)[0][0]
        
        print(f"✓ Test prediction: {prediction}")
        print("✓ ANN model loading test successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing saved model: {e}")
        return False

if __name__ == "__main__":
    success = create_ann_model()
    if success:
        print("\n🎉 SUCCESS! Your ANN model is ready for deployment!")
    else:
        print("\n❌ Something went wrong.")
