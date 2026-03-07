from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

from sklearn.preprocessing import StandardScaler

application=Flask(__name__)

app=application

##Route for a home page


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        reading_score_val = request.form.get('reading score')
        writing_score_val = request.form.get('writing score')

        if not reading_score_val or not writing_score_val:
            return render_template('home.html', results="Please enter both scores.")

        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(reading_score_val),
            writing_score=float(writing_score_val),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        pred_pipeline=PredictPipeline()
        results=pred_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")