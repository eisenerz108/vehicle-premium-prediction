from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_premium():
    driverAge = int(request.form.get("driverAge"))
    driverExperience = int(request.form.get("driverExperience"))
    previousAccidents = int(request.form.get("previousAccidents"))
    annualMileage = int(request.form.get("annualMileage"))
    carManufacturingYear = int(request.form.get("carManufacturingYear"))
    carAge = int(request.form.get("carAge"))

    #Prediction
    result = model.predict(np.array([driverAge,
                                     driverExperience,
                                     previousAccidents,
                                     annualMileage,
                                     # carManufacturingYear,
                                     carAge]).reshape(1, -1))
    return str(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

