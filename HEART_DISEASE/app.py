import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("hmodel.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    result = 'present' if prediction == 1 else 'not present'

    # Pass the result to the template
    return render_template("index.html", prediction_text=f"Disease is {result}")
    #return render_template("index.html", prediction_text = "Disease is  {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)