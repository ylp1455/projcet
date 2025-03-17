from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import joblib

app = Flask(__name__)

# Load saved preprocessing objects
try:
    le_grade = joblib.load('grade_encoder.joblib')
    scaler = joblib.load('feature_scaler.joblib')
except FileNotFoundError:
    print("Preprocessing files not found. Creating new ones...")
    le_grade = LabelEncoder()
    scaler = MinMaxScaler()

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get parameters safely
        grade = request.args.get('grade')
        time_taken = request.args.get('time_taken')
        
        # Validate inputs
        if not grade or not time_taken:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Convert to appropriate types
        current_grade = int(grade)
        time_taken = float(time_taken)
        
        # Prepare data for prediction
        data = {
            'current_grade': [current_grade],
            'time_taken': [time_taken]
        }
        df = pd.DataFrame(data)
        
        # Encode grade
        df['encoded_grade'] = le_grade.fit_transform(df['current_grade'])
        
        # Scale features
        scaled_features = scaler.fit_transform(df[['time_taken', 'encoded_grade']])
        
        # Determine grade adjustment based on time taken
        # Adjust grade up if time is less than 60 seconds
        # Adjust grade down if time is more than 90 seconds
        if time_taken < 60:
            adjusted_grade = min(current_grade + 1, 10)  # Cap at grade 10
        elif time_taken > 90:
            adjusted_grade = max(current_grade - 1, 1)   # Minimum grade 1
        else:
            adjusted_grade = current_grade
            
        return jsonify({
            'status': 'success',
            'adjusted_grade': adjusted_grade,
            'adjustment': adjusted_grade - current_grade,
            'input_data': {
                'original_grade': current_grade,
                'time_taken': time_taken
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)