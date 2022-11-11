from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():
    M01AB=float(request.form['M01AB'])
    M01AE= float(request.form['M01AE'])
    N02BA= float(request.form['N02BA'])
    N02BE = float(request.form['N02BE'])
    N05B = float(request.form['N05B'])
    N05C= float(request.form['N05C'])
    R03= float(request.form['R03'])
    R06= float(request.form['R06'])
    Year= float(request.form['Year'])
    Month = float(request.form['Month'])
    Hour = float(request.form['Hour'])
    Weekday_Name = float(request.form['Weekday_Name'])

    X= np.array([[M01AB,M01AE,N02BA,N02BE,N05B,N05C,R03,R06,Year,Month,Hour,Weekday_Name ]])

    scaler_path= r'"C:\Users\RAJU RANJAN\Desktop\Sales_forecasting\models\sc.sav"'

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)

    model_path=r'"C:\Users\RAJU RANJAN\Desktop\Sales_forecasting\models\rf.sav"'

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)

    return jsonify({'Prediction': float(Y_pred)})

if __name__ == "__main__":
    app.run(debug=True, port=9457)
