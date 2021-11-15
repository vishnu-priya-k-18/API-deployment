import os

import storage as storage
from flask import Flask, request, app
import pickle
import numpy as np
import requests

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ilab-match-predictor-a7db67a95d7a.json" # change for your GCP key
PROJECT = "ilab-match-predictor" # change for your GCP project
REGION = "us-central1"

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_FILENAME = os.environ['MODEL_FILENAME']
MODEL = None

@app.before_first_request
def _load_model():
    global MODEL
    client = storage.Client()
    bucket = client.get_bucket(MODEL_BUCKET)
    blob = bucket.get_blob(MODEL_FILENAME)
    s = blob.download_as_string()

    MODEL = pickle.loads(s)

app = Flask(__name__, template_folder='templates')

#model = pickle.load(open('model.pkl', 'rb'))  # load


@app.route('/', methods=['GET', 'POST'])
def getResult():
    if request.method == "GET":
        url = "http://127.0.0.1:5000/"

        data = {

            "experience": 5
        }

        r = requests.post(url, json=data)

        return str(r.json())

    if request.method == "POST":
        data = request.get_json()
        experienceOfEmployees = data["experience"]
        user_input = np.array([[experienceOfEmployees]])
        prediction = MODEL.predict(user_input)

        return str(prediction)


if __name__ == "__main__":
    app.debug = True
    app.run()
