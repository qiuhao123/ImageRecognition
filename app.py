#!/usr/bin/env python
from flask import Flask, render_template, request, url_for,redirect
#from werkzeug import secure_filename
import boto3
import os

#ml package
import pandas as pd
import numpy
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model 
import warnings
warnings.filterwarnings('ignore')

#define a flask app
app = Flask(__name__)

BUCKET = 'image-recognition2020'
UPLOAD_FOLDER = 'uploads'
MODEL_FOLER = 'model'

model = ResNet50(weights='imagenet')
model.save(MODEL_FOLER+"/resnet.h5")
print("Saved model to disk")
    

def predict_image(image):
    model = load_model('model/resnet.h5')
    image_array = img_to_array(image)
    image_prepared = preprocess_input(image_array)
    yhat = model.predict(image_prepared)
    label = decode_predictions(yhat)
    return label[0][0]
    
#the upload_file function takes a file name and upload that file to a specific bucket
def upload_file(file_name,bucket):
    object_name = file_name
    client = boto3.client('s3')
    response = client.upload_file(file_name,bucket,object_name)
    
    return response

#the download file function takes in a file and bucket and download it to the file we specify
def download_file(file_name,bucket):
    client = boto3.resource('s3')
    output = f"downloads/{file_name}"
    client.Bucket(bucket).download_file(file_name, output)
    return output

#we will get a list of file name that is in the bucket   
def list_files(bucket):
    s3 = boto3.client('s3')
    contents = []
    for item in s3.list_objects(Bucket=bucket)['Contents']:
        contents.append(item)
    return contents


@app.route('/')
def index():
    #save_model()
    return render_template('index.html')

@app.route("/storage")
def storage():
    contents = list_files("image-recognition2020")
    return render_template('storage.html', contents=contents)
    
@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        s3 = boto3.resource('s3')
        f = request.files['file']
        client = boto3.client('s3')
        f.save(os.path.join(UPLOAD_FOLDER, f.filename))
        response = upload_file(f"uploads/{f.filename}", BUCKET)
        image = load_img(f"uploads/{f.filename}",target_size=(224,224))
        result = predict_image(image)
        return str(result)


if __name__ == '__main__':
    app.run(debug=True, port=8080, host = '0.0.0.0')