from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os


app = Flask(__name__)

# Expected raw input fields
REQUIRED_FIELDS = [
    'age','job','marital','education','default','housing','loan',
    'contact','month','day_of_week','campaign','pdays','previous','poutcome',
    'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'
]

# Load artifacts (saved previously)
try:
    ART = joblib.load('preprocessor.pkl')  # {'preprocessor','high_skew_cols','numeric_cols'}
    PREPROCESSOR = ART['preprocessor']
    HIGH_SKEW_COLS = ART['high_skew_cols']
    NUMERIC_COLS = ART['numeric_cols']
    MODEL = joblib.load('model.pkl')
    print('Artifacts loaded successfully!')
except FileNotFoundError:
    PREPROCESSOR = None
    HIGH_SKEW_COLS = []
    NUMERIC_COLS = []
    MODEL = None
    print('Artifact files not found. Please train and save artifacts first.')


@app.route('/')
def home():
    return jsonify({
        'message': 'Bank Marketing Subscription Prediction API',
        'endpoints': {
            '/predict': 'POST - Send customer data to get prediction',
            '/health': 'GET - Health check',
            '/features': 'GET - Required fields and example payload'
        },
        'required_fields': REQUIRED_FIELDS
    })


@app.route('/features', methods=['GET'])
def features():
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
    return jsonify({'features': REQUIRED_FIELDS, 'example_payload': example})


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None or PREPROCESSOR is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            payload = [data]
        else:
            payload = data

        df_in = pd.DataFrame(payload)
        # Ensure all required fields present
        for col in REQUIRED_FIELDS:
            if col not in df_in.columns:
                df_in[col] = np.nan
        df_in = df_in[REQUIRED_FIELDS]

        # Apply saved skew transform on numeric columns
        for c in NUMERIC_COLS:
            if c in HIGH_SKEW_COLS:
                min_val = df_in[c].min()
                if pd.notna(min_val) and min_val < 0:
                    df_in[c] = df_in[c] - min_val
                df_in[c] = np.log1p(df_in[c].clip(lower=0))

        X_proc = PREPROCESSOR.transform(df_in)
        if hasattr(MODEL, 'predict_proba'):
            probs = MODEL.predict_proba(X_proc)[:, 1]
        else:
            # fallback for models without predict_proba
            preds_raw = MODEL.predict(X_proc)
            probs = np.asarray(preds_raw).astype(float).ravel()
        preds = (probs >= 0.5).astype(int)

        return jsonify({
            'success': True,
            'prediction': preds.tolist(),
            'probability': probs.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': MODEL is not None})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)


