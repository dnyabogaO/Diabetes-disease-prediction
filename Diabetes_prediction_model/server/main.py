
import pickle
from flask import Flask, request, jsonify, render_template
from model_files.ml_model import predict_diab

app = Flask("diabetes_prediction", template_folder='../templates')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    diabetis_config = request.get_json()

    with open('../model_files/diabetis_predict_model.pkl', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_diab(diabetis_config, model)
    result = {
        'diabetes_prediction': list(predictions)
    }
    return jsonify(result)


# @app.route('/', methods = ['GET'])
# def ping():
#     return "Pinging model Application!!"


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port=9696)