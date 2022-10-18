from flask import Flask,render_template,redirect,url_for,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route("/")
def home():
    return render_template("pizza.html")
@app.route("/predict",methods=["post"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    pred=model.predict(features)
    if pred==1:
        strr="your can eat pixzza"
    else:
        strr="you cannot eat pizza"
    return render_template("pizza.html",text_prediction="your model output is{} and {}".format(pred,strr))
    

if __name__=='__main__':
    app.run()

