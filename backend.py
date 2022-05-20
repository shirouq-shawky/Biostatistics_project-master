import numpy as np
from crypt import methods
from pyexpat import model
from flask import Flask, app, render_template, url_for, request
import pickle

def pre_process(x):
    
    
    
    return x


def load_model(x):
    model= pickle.load(open('model.py', 'rb'))
    out= model.predict(np.array(x), np.reshape(1, -1))
    return out(0)

app = Flask(__name__)
submition=[]


@app.route('/', methods=["GET"])
def inder():
    return render_template('index1.html')

@app.route('/predict', methods=['GET', 'POST'])

def home():
    if request.method== "POST":
        print(request.form)
        submition.append(
                ( 
             int(request.form.get("age")),
             float(request.form.get("amount")),
             request.form.get("gender"),
             float(request.form.get("height")),
             float(request.form.get("weight")),
             int(request.form.get("AP-HI")),
             int(request.form.get("AP-HO")),
             int(request.form.get("Cholesterol")),
             int(request.form.get("Glucose")),
             request.form.get("smoke"),
             request.form.get("alcohol"),
             request.form.get("Cardio"),
             request.form.get("Active"),
                 )
                       )
    return render_template("index1.html")


@app.route("/submition")
def show_submition():
    return render_template("submition.jinja2" ,entries=submition)



def predict():
    if request == "POST":
        f = request.file['file']
        x = pre_process(f)
        out = load_model(x)
        if out == 0 :
            return "Negative"
        else:
            return "Positive"
    return None



if __name__  == '__main__' :
    app.run(debug=True)