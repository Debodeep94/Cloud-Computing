from flask import Flask, render_template, request
import numpy as np
import PIL.Image as Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
#import tensorflow_hub as hub

application = Flask(__name__)

image_size = (180, 180)
model = keras.models.load_model('xceptionv2_119_1.000.h5')

labels = {
    'fire_image',
    'non_fire_image'
}

def pred_image(location):
    image=load_img(location, target_size=image_size)
    x = np.array(image)
    X = np.array([x])
    X = tf.keras.applications.xception.preprocess_input(X)
    preds = model.predict(X)
    #print(preds)
    predicted_class = preds[0]
    return predicted_class

# routes
@application.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@application.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"

@application.route("/submit", methods = ['GET', 'POST'])

def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename	
        
        img.save(img_path)
        p=pred_image(img_path)
        print('Image Class: {0}, Prediction: {1}'.format(list(labels), p))
        print(p.argmax())
    return render_template("index.html",image_class=list(labels)[p.argmax()],prediction=max(p), img_path = img_path)


if __name__ =='__main__':
    #app.debug = True
    application.run(host='0.0.0.0', port= 8080)
