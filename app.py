# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import os, os.path
#from keras.preprocessing.image import load_img
#from keras.models import load_model
import skimage
from skimage.metrics import structural_similarity
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO 
from werkzeug.utils import secure_filename 
from skimage.transform import resize


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
 
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
model = tf.keras.models.load_model('model.h5')
modelt = tf.keras.models.load_model('modelt1.h5')

#dir_path = '/content/drive/MyDrive/thyroid_images/test'
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/gettrtype',methods=['POST'])
def gettrtype():
    output = ''
    file_pic = request.files['file']
    img = Image.open(file_pic)
    newsize = (200, 200)
    img = img.resize(newsize)
    #img = image.resize((200, 200))
    # plt.imshow(img)
    # plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    #print(images)
    val = modelt.predict(images)
    predcls = np.argmax(val, axis=1)
    if predcls == 0:
      output = 'This image is tr1'
    if predcls == 1:
      output = 'This image is tr2'
    if predcls == 2:
      output = 'This image is tr3'
    if predcls == 3:
      output = 'This image is tr4'
    if predcls == 4:
      output = 'This image is tr5'

    return render_template('tirad.html', output = output)


@app.route('/getprediction',methods=['POST'])
def getprediction():
    output = ''
    file_pic = request.files['file']
    img = Image.open(file_pic)
    newsize = (200, 200)
    img = img.resize(newsize)
    #img = image.resize((200, 200))
    # plt.imshow(img)
    # plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    #print(images)
    val = model.predict(images)
    if val == 0:
      output = 'This image is Benign'
    else:
      output = 'This image is Malign'

    return render_template('benignormalign.html', output = output)

@app.route('/tirad')
def tirad():
    return render_template('tirad.html')


#Works well with images of different dimensions
def orb_sim(img1, img2):
  # SIFT is no longer available in cv2 so using ORB
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]
  #print(similar_regions)  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)
   

def structural_sim(img1, img2):
    
  sim, diff = structural_similarity(img1, img2, full=True)
  return sim

#manual testing
@app.route('/gettrtypeorb',methods=['POST'])
def gettrtypeorb():
      output = ''
      output1 = ''
      values=[]
      values_ssm=[]
      input_file = request.files['file']
      # store in folder and get a file
      path = f"static/inputdata"
      #if os.path.exists(path):
      #       os.rmdir(path) 
      if not os.path.exists(path):
          os.makedirs(path)
      app.config['UPLOAD_FOLDER'] = path
      filename = secure_filename(input_file.filename)
      input_file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

      get_url = path + '/' + filename
      img1 = cv2.imread(get_url,0)
      print('img1',img1)
      #print(input_file)
      url1 = f"static/tr1"
      url2 = f"static/tr2"
      url3 = f"static/tr3"
      url5 = f"static/tr5"
      dirs1 = os.listdir(url1)
      dirs2 = os.listdir(url2)
      dirs3 = os.listdir(url3)
      dirs5 = os.listdir(url5)
      tr1_orb = 0
      tr2_orb = 0
      tr3_orb = 0
      tr5_orb = 0
      tr1_ssm = 0
      tr2_ssm = 0
      tr3_ssm = 0
      tr5_ssm = 0
      #print(dirs1)
      for file in dirs1:
            file1 = Image.open(os.path.join(url1,file))
            print(file1)
            #img1 = cv2.imread(input_file,0)
            img2 = cv2.imread('static/tr1/tr1.jpg',0)
            img5 = resize(img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)
            
            print(img2)
            tr1_orb = orb_sim(img1,img2)
            tr1_ssm = structural_sim(img1,img5)
            values.append(tr1_orb)
            values_ssm.append(tr1_ssm)
            #print(type(file1))
      for file in dirs2:
            file2 = Image.open(os.path.join(url2,file))
            #img1 = cv2.imread('static/tr2/tr2.jpg',0)
            img2 = cv2.imread('static/tr2/tr2.jpg',0)
            img5 = resize(img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)

            tr2_orb = orb_sim(img1,img2)
            tr2_ssm = structural_sim(img1,img5)
            values.append(tr2_orb)
            values_ssm.append(tr2_ssm)
            #print(type(file2))
      for file in dirs3:
            file3 = Image.open(os.path.join(url3,file))
            #img1 = cv2.imread(input_file_img,0)
            img2 = cv2.imread('static/tr3/tr3.jpg',0)
            img5 = resize(img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)

            tr3_orb = orb_sim(img1,img2)
            tr3_ssm = structural_sim(img1,img5)
            values.append(tr3_orb)
            values_ssm.append(tr3_ssm)
            #print(type(file3))
      for file in dirs5:
            file5 = Image.open(os.path.join(url5,file))
            #img1 = cv2.imread(input_file_img,0)
            img2 = cv2.imread('static/tr5/tr5.jpg',0)
            img5 = resize(img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)
            
            tr5_orb = orb_sim(img1,img2)
            tr5_ssm = structural_sim(img1,img5)
            values.append(tr5_orb)
            values_ssm.append(tr5_ssm)
            #print(type(file5))
      print(values)
      print(values_ssm)
      max_values = max(values)
      max_values_ssm = max(values_ssm)
      if max_values == tr1_orb:
            output = "This scanned image is tr1 type using orb.."
      if max_values == tr2_orb:
            output = "This scanned image is tr2 type using orb.."
      if max_values == tr3_orb:
            output = "This scanned image is tr3 type using orb.."
      if max_values == tr5_orb:
            output = "This scanned image is tr5 type using orb.."
      if max_values_ssm == tr1_ssm:
            output1 = "This scanned image is tr1 type using structural similarity"
      if max_values_ssm == tr2_ssm:
            output1 = "This scanned image is tr2 type using structural similarity"
      if max_values_ssm == tr3_ssm:
            output1 = "This scanned image is tr3 type using structural similarity"
      if max_values_ssm == tr5_ssm:
            output1 = "This scanned image is tr5 type using structural similarity"


      return render_template('orb.html',output = output,output1=output1)


@app.route('/benignormalign')
def benignormalign():
    return render_template('benignormalign.html')

@app.route('/gettiradorb')
def gettiradorb():
    return render_template('orb.html')

@app.route('/gettiradssm')
def gettiradssm():
    return render_template('ssim.html')

 
# main driver function
if __name__ == '__main__':
    app.run(debug=True)