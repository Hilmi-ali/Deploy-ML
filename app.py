import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('iris_knn.pkl')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/coba')
def testing():
    return "Hello Coba Flask"

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        strfeatures = [X for X in request.form.values()]
        strfeatures = [x for x in strfeatures if x != '']
        float_features = [[float(y) for y in strfeatures]]
        prediction = model.predict(float_features)[0]
        if prediction == 0:
            return render_template("index.html", prediction_text="0. Setosa")
        elif prediction == 1:
            return render_template("index.html", prediction_text="1. Versicolor")
        elif prediction == 2:
            return render_template("index.html", prediction_text="2. Virginica")
        else:
            return render_template("index.html", prediction_text="Error Classification")
    else:
        return render_template("index.html", prediction_text="")
    
if __name__ == "__main__":
    app.run(debug=True)
