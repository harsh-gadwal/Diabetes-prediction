# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the model, scaler, and encoders
# with open('diabetes_model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# with open('scaler.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)

# with open('gender_encoder.pkl', 'rb') as gender_file:
#     gender_encoder = pickle.load(gender_file)

# with open('smoking_encoder.pkl', 'rb') as smoking_file:
#     smoking_encoder = pickle.load(smoking_file)

# # Route to serve the HTML page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # API route for predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     print(data)
#     try:
#         # Extract and encode categorical inputs
#         gender = gender_encoder.transform([data['gender'].strip()])[0]
#         print("gender: ", gender)
#         smoking_history = smoking_encoder.transform([data['smoking_history']])[0]
#         print("smoking_history", smoking_history)
#         # Extract numeric inputs
#         # gender = 1
#         # smoking_history=1
#         age = float(data['age'])
#         print(age)
#         hypertension = int(data['hypertension'])
#         print(hypertension)

#         heart_disease = int(data['heart_disease'])
#         print(heart_disease)

#         bmi = float(data['bmi'])
#         print(bmi)
#         HbA1c_level = float(data['HbA1c_level'])
#         print(HbA1c_level)
#         blood_glucose_level = float(data['blood_glucose_level'])
#         print(blood_glucose_level)
#     except (KeyError, ValueError) as e:
#         return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

#     # Prepare input features
#     input_features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
#     print(input_features)
#     # Scale the input features
#     input_features_scaled = scaler.transform(input_features)
#     print(input_features_scaled)
#     # Make a prediction
#     prediction = model.predict(input_features_scaled)
#     print(prediction)
#     print(prediction[0])
#     # Return the result
#     result = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'
#     return jsonify({'prediction': result})

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and encoders
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('gender_encoder.pkl', 'rb') as gender_file:
    gender_encoder = pickle.load(gender_file)

with open('smoking_encoder.pkl', 'rb') as smoking_file:
    smoking_encoder = pickle.load(smoking_file)

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract and encode categorical inputs
        gender = gender_encoder.transform([data['gender'].strip()])[0]
        smoking_history = smoking_encoder.transform([data['smoking_history']])[0]

        # Extract numeric inputs
        age = float(data['age'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heart_disease'])
        bmi = float(data['bmi'])
        HbA1c_level = float(data['HbA1c_level'])
        blood_glucose_level = float(data['blood_glucose_level'])

    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

    # Prepare input features
    input_features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
    input_features_scaled = scaler.transform(input_features)
    
    # Make a prediction
    prediction = model.predict(input_features_scaled)

    # Return the result
    result = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
    