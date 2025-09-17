from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

try:
    # Load the model (H5 format)
    model = tf.keras.models.load_model('tensorflow_model.h5')
    print("TensorFlow model loaded successfully!")
    
    preprocessor = joblib.load('preprocessor.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError:
    print("Model files not found. Please run save_tensorflow_model.py first.")
    model = None
    preprocessor = None
    feature_names = []



@app.route('/')
def home():
    return jsonify({
        'message': 'Bank Marketing Subscription Prediction API',
        'endpoints': {
            '/predict': 'POST - Send customer data to get prediction',
            '/health': 'GET - Health check',
            '/features': 'GET - Required fields and example payload'
        },
        'required_fields': ['age','job','marital','education','default','housing','loan',
    'contact','month','day_of_week','campaign','pdays','previous','poutcome',
    'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for Bank Marketing Subscription Prediction"""
    if model is None or preprocessor is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        data = request.get_json()
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
        
        # Reorder columns to match training order
        input_df = input_df[feature_names]
        
        # Apply preprocessing
        X_processed = preprocessor.transform(input_df)
        
        # Get prediction from TensorFlow model
        prediction_proba = model.predict(X_processed, verbose=0)[0][0]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(prediction_proba),
            'message': f'Predicted Bank Marketing Subscription: {"Yes" if prediction == 1 else "No"}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })



@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/features', methods=['GET'])
def features():
    """Get required features and example payload"""
    example = {
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
    }
    return jsonify({
        'features': feature_names if feature_names else ['age','job','marital','education','default','housing','loan',
        'contact','month','day_of_week','campaign','pdays','previous','poutcome',
        'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],
        'example_payload': example
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
    
    


