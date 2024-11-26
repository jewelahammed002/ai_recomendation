from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the model and encoders
model = joblib.load("restaurant_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    location = request.form['location']
    cuisine = request.form['cuisine']
    price_range = request.form['price_range']
    mood = request.form['mood']

    # Preprocess input using encoders
    try:
        input_data = pd.DataFrame({
            'Location': [label_encoders['Location'].transform([location])[0]],
            'Type of Cuisine': [label_encoders['Type of Cuisine'].transform([cuisine])[0]],
            'Price Range': [label_encoders['Price Range'].transform([price_range])[0]],
            'Mood': [label_encoders['Mood'].transform([mood])[0]]
        })

        # Make prediction
        prediction = model.predict(input_data)
        restaurant = label_encoders['Restaurant'].inverse_transform(prediction)[0]

        return render_template('index.html', prediction=restaurant)

    except ValueError:
        return render_template('index.html', error="Invalid input. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)