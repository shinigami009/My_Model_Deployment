from flask import Flask,render_template,request
import pickle as pkl
import pandas as pd
import numpy as np
########################################
app=Flask(__name__)
with open('App\model_pkl', 'rb') as pickle_file:
    model = pkl.load(pickle_file)
with open('App\encoder_cp', 'rb') as pickle_file:
    encoder_cp=pkl.load(pickle_file)
with open('App\encoder_exang', 'rb') as pickle_file:
    encoder_exang=pkl.load(pickle_file)
with open('App\encoder_sex', 'rb') as pickle_file:
    encoder_sex=pkl.load(pickle_file)
with open('App\encoder_slope', 'rb') as pickle_file:
    encoder_slope=pkl.load(pickle_file)
########################################
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/action_page.php",methods=['POST'])
def predict():
    input_features = [x for x in request.form.values()]
    if(input_features[1]=='non-anginal pain' or input_features[1]=='typical angina'):
        input_features[1]=" "+input_features[1]
    input_features[4]=int(input_features[4])
    input_features[5]=int(input_features[5])
    df=pd.DataFrame([np.array(input_features).transpose()],columns=['sex', 'cp', 'exang', 'slope', 'ca', 'thal'])
    df.sex=encoder_sex.transform(df.sex)
    df.cp=encoder_cp.transform(df.cp)
    df.exang=encoder_exang.transform(df.exang)
    df.slope=encoder_slope.transform(df.slope)
    result=model.predict(df)
    print(result)
    #print(encoder.transform(input_features))
    return render_template("index.html",result=result[0])
#######################################
if __name__=='__main__' :
    app.run(debug=True)
