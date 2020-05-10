from flask_cors import CORS
from flask import Flask,request,send_file,send_from_directory
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import json

BackEndApp = Flask(__name__)
cors = CORS(BackEndApp, resources={r"/*": {"origins": "*"}})

pkl_filename = "cactus_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

@BackEndApp.route("/")
def main():
    return "Hello"


@BackEndApp.route("/predict/",methods=['GET'])
def predictOneResult():
    FT = str(request.args.get("ft"))
    data = []
    for i in range(13):
        data.append(int(FT[i]))
    
    predict_result = model.predict([data])
    result = {
        "Result" : predict_result[0]
    } 
    return json.dumps(result)

if __name__ == '__main__':
    BackEndApp.run(port=80)