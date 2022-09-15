import json
import boto3
import numpy as np
import PIL.Image as Image

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub



image_size = (180, 180)
model = keras.models.load_model('xceptionv2_119_1.000.h5')

#model.build([None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

#imagenet_labels= np.array(open('model/ImageNetLabels.txt').read().splitlines())
s3 = boto3.resource('s3')

labels = {
    'fire_images',
    'non_fire_images'
}

def lambda_handler(event, context):
  bucket_name = event['Records'][0]['s3']['bucket']['name']
  key = event['Records'][0]['s3']['object']['key']

  img = readImageFromBucket(key, bucket_name).resize(image_size)
  x = np.array(img)
  X = np.array([x])
  X = tf.keras.applications.xception.preprocess_input(X)

  preds = model.predict(X)
  print(f'Image file name: {key}')

  print('ImageName: {0}, Prediction: {1}'.format(labels, preds[0]))

def readImageFromBucket(key, bucket_name):
  bucket = s3.Bucket(bucket_name)
  object = bucket.Object(key)
  response = object.get()
  return Image.open(response['Body'])