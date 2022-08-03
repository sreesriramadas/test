import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
pickle_in = open("model.pkl","rb")
random = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = random.predict(final_features)
    return render_template('index.html', prediction_text = 'The fish belongs to species {}'.format(str(prediction)))

if __name__=='__main__':
    app.run()