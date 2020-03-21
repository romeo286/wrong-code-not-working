

from flask import Flask

from flask import render_template, redirect, url_for, request

from sklearn.externals import joblib
import pandas as pd 
import numpy as np


app = Flask (__name__)




@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['GET', 'POST']) 

def predict():
    if request.method == 'POST':
       try:
           x_box = str(request.form['x_box'])
        #    ybox width high onpix x-bar y-bar x2bar y2bar xybar x2ybr xy2br x-ege xegvy y-ege yegvx
           ybox = str(request.form['ybox'])
           width = str(request.form['width'])
           high = str(request.form['high'])
           onpix = str(request.form['onpix'])
           x_bar= str(request.form['x_bar'])
           y_bar = str(request.form['y_bar'])
           x2bar = str(request.form['x2bar'])
           y2bar = str(request.form['y2bar'])
           xybar = str(request.form['xybar'])
           x2ybr = str(request.form['x2ybr'])
           x_ege = str(request.form['x_ege'])
           xegvy= str(request.form['xegvy'])
           y_ege = str(request.form['y_ege'])
           yegvx = str(request.form['yegvx'])
           pred_args =[x_box ,ybox, width ,high ,onpix ,x_bar ,y_bar ,x2bar ,y2bar ,xybar ,x2ybr ,x_ege ,xegvy ,y_ege ,yegvx]
           pred_args_arr = np.array( pred_args)
           pred_args_arr = pred_args_arr.reshape(1,-1)
           rfc = open ("random_forest_prototype.pkl","rb")
           ml_model = joblib.load(rfc)
           model_prediction = ml_model.predict(pred_args_arr)
          
       
        
          

       except ValueError:
           return "Please check the following inputs  " 

    return render_template('predict.html', prediction = model_prediction)



if __name__ == '__main__':
      app.debug=True
      app.run()