from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os


app = Flask(__name__)

try:
    model = joblib.load('Bank_model.joblib')
    feature_names = joblib.load('Bank_feature_names.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Please train the model first.")
    model = None
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
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        data = request.get_json()
        
        input_df = pd.DataFrame([data])
        
       
        
        missing_features = set(feature_names) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
        
        input_df = input_df[feature_names]
        
        
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'message': f'Predicted Bank Marketing Subscription: {prediction:.2f} mg/dL'
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
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
    
    


