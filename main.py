from flask import Flask, request
import pickle
import numpy as np
import requests

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('model.pkl', 'rb'))  # load


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
        prediction = model.predict(user_input)

        return str(prediction)


if __name__ == "__main__":
    app.debug = True
    app.run()
