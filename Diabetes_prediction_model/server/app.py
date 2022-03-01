from flask import Flask, render_template, request
from flask import jsonify
#import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__, template_folder='../templates')
model = pickle.load(open('../model_files/diabetis_predict_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose=int(request.form['Glucose'])
        BloodPressure=int(request.form['BloodPressure'])
        SkinThicknes=int(request.form['SkinThicknes'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        prediction=model.predict([[Pregnancies,Glucose,BloodPressure,SkinThicknes,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        output=round(prediction[0],2)
        if output<=0:
            return render_template('index.html',prediction_texts="No Diabetes Detected {}".format(output))
        else:
            return render_template('index.html',prediction_text="Diabetes Detected {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)