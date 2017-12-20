from keras.optimizers import SGD
import tensorflow as tf
import cv2
import requests
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from flask import Flask
from densenet121 import DenseNet
weights_path = 'densenet121_weights_tf.h5'

app = Flask(__name__)

global model, graph
# initialize these variables
model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
graph = tf.get_default_graph()
classes = []
with open('resources/classes.txt', 'r') as list_:
    for line in list_:
        classes.append(line.rstrip('\n'))
#
# @app.route('/')
# def index():
#     # initModel()
#     # render out pre-built HTML file right on the index page
#     with open('G:\programmer\python\PycharmProjects\PycharmProjects\saver_model\index.html', 'r') as output:
#         return render_template(output)
def deal_image(url):

  response = requests.request('get',url)
  img = np.array(bytearray(response.content))
  img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
  # img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
  image_path='resources/'
  cv2.imwrite(os.path.join(image_path, "4.jpg"), img)
  im = cv2.resize(cv2.imread(image_path+"4.jpg"),(224, 224)).astype(np.float32)

  # im = cv2.resize(cv2.imread('shark.jpg'), (224, 224)).astype(np.float32)

  # Subtract mean pixel and multiple by scaling constant
  im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
  im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
  im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017

  im = np.expand_dims(im, axis=0)
  return im

@app.route('/predict/<path:url>', methods=['GET', 'POST'])
def predict(url):
    img = deal_image(url)  # image array
    # resized_image = img.reshape(1, 32, 32, 3)
    print("debug2")

    with graph.as_default():
        out = model.predict(img)
        response=str(classes[np.argmax(out)])
        print('Prediction: %s' %response )
        return response

if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 8000))
    # run the app locally on the givn port
    app.run(host='127.0.0.1', port=port)
    # optional if we want to run in debugging mode
    # app.run(debug=True)
