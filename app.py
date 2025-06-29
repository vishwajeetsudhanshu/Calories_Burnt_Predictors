from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load("calories_burnt_model_pipeline.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None

    if request.method == 'POST':
        
        gender = request.form['gender']
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        activity = request.form['activity']
        duration = float(request.form['duration'])

        input_data = pd.DataFrame([{
            'Gender': gender,
            'Age': age,
            'Height (cm)': height,
            'Weight (kg)': weight,
            'Activity': activity,
            'Duration (minutes)': duration
        }])

        # Predict using the pipeline model
        prediction = model.predict(input_data)
        result = round(prediction[0], 2)

    return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
