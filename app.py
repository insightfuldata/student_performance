import joblib as jb
import numpy as np
from flask import Flask, request, jsonify

#Load the model
model = jb.load("Student_Performance_Model.pkl")

#Initialize Flask app
app = Flask(__name__)

#Define the endpoints
@app.route("/", methods = ["GET"])
def home():
    return "Student Performance App is running"

@app.route("/predict", methods = ["POST"])
def predict():
    try:
        data = request.get_json(force = True)
        study_hours = np.array(data['study_hours']).reshape(1, -1)
        prediction = model.predict(study_hours)[0]
        return jsonify({f"Predicted GPA for {study_hours[0][0]} hours is": np.round(prediction,2)})
    except Exception as e:
        return jsonify({"error occured": str(e)})
if __name__ == "__main__":
    app.run(debug = True)