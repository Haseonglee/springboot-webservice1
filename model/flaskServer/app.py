import os
from IPython.core import magic
from flask import Flask, jsonify, request, render_template
from flask_restful import Api, Resource
from scipy import misc
from tensorflow.keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras_applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf


app = Flask(__name__)
model = None

# @app.route("/")
# @app.route("/index")
# def index():
#     return render_template('index.html')
def load_model():
    global model
    model = load_model('C:/Users/175767/PycharmProjects/springboot-webservice1/model/flaskServer/model/COVIDMD.h5');
    global graph
    graph = tf.get_default_graph()


def prepare_image(image, target):
    # image mode should be "RGB"
    if image.mode != "RGB":
        image = image.convert("RGB");

    # resize for model
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return it
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # view로부터 반환될 데이터 딕셔너리를 초기화
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target=(112, 112))

            with graph.as_default():

                preds = model.predict(image)
                results = imagenet_utils.decode_predictions(preds)
                data["predictions"] = []

                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                data["success"] = True

    return flask.jsonify(data)

    # if request.method == "POST" and request.files.get("image"):
    #     image = request.files["image"].read()
    #     extention = magic.from_buffer(image).split()[0].upper()
    #     if extention == 'GIF':
    #         imageObject = Image.open(io.BytesIO(image))
    #         imageObject.seek(0)
    #         imageObject = imageObject.convert('RGB')
    #         image = np.array(imageObject)
    #     elif extention != 'JPEG' and extention != 'PNG':
    #         return jsonify(data)
    #     else:
    #         image = Image.open(io.BytesIO(image)).convert('RGB')

        # 입력 이미지를 분류하고 클라이언트로부터 반환되는 예측치들의 리스트를 초기화 합니다.


        # 결과를 반복하여 반환된 예측 목록에 추가
        # for (imagenetID, label, prob) in results[0]:
        #     r = {"label": label, "probability": float(prob)}
        #     data["predictions"].append(r)
        # 요청 성공

        # JSON 형식으로 데이터 딕셔너리 반환

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()
