#!/usr/bin/env python
from flask import Flask, render_template, request, url_for,redirect
#from werkzeug import secure_filename
import boto3
import os

#ml package
import pandas as pd
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model 
from keras.models import model_from_json


import warnings
warnings.filterwarnings('ignore')

#define a flask app
app = Flask(__name__)

BUCKET = 'image-recognition2020'
MODEL_BUCKET = 'model-bucket2020'
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/resnet.h5'




# model = ResNet50(weights='imagenet')
# # Convert your existing model to JSON
# saved_model = model.to_json()

# # Write JSON object to S3 as "keras-model.json"
# client = boto3.client('s3')
# client.put_object(Body=saved_model,
#                   Bucket=MODEL_BUCKET,
#                   Key='keras-model.json')


# Read the downloaded JSON file
s3 = boto3.client('s3')
s3.download_file(MODEL_BUCKET,"keras-model.json" , "keras-model.json")
with open('keras-model.json', 'r') as model_file:
   loaded_model = model_file.read()

# Convert back to Keras model
model = model_from_json(loaded_model)
# model = load_model(MODEL_PATH)
# #for threading purposes
model._make_predict_function()
    
def model_predict(img_path,model):
    img = load_img(img_path,target_size=(224,224))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds
    
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
        file_path = f"uploads/{f.filename}"
        response = upload_file(file_path, BUCKET)
       
        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True, port=8080, host = '0.0.0.0')