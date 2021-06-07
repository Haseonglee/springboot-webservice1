import os

from IPython.core import magic
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
import numpy as np
import io
import cv2


app = Flask(__name__)
model = None
api = Api(app)

@app.route("/")
def index():
    html = '''
<html>
<head>
    <title>물체 분류</title>
</head>
<body>
    <center>
    물체 분류<br>
    <form action="/predict" method="post" enctype="multipart/form-data" id="form">
        이미지 파일 <input type="file" id="uploadFile"><br>
        <input type="submit" value="분류">
    </form>
    <span id="result">분류 결과</span> 
    </center>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
    <script type="text/javascript">
$("#form").submit(function(event) {
    event.preventDefault(); //폼 제출 방지

    var formData = new FormData();
    formData.append("file", $("#uploadFile").prop("files")[0]);
    $.ajax({
        type: "post",
        url: "/predict",
        data: formData, 
        contentType: false,
        processData: false,
        success: function(data){
            $("#result").html(data);
        }
    });
});      
    </script>        
</body>
</html>
'''

    return html

def load_model():
    image_dir = '/content/drive/MyDrive/Covid Model/test/COVID/'
    categories = ['COVID', 'Non-COVID']

    def Dataization(img_path):
        image_w = 112
        image_h = 112
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
        return (img / 256)

    src = []
    name = []
    test = []

    for file in os.listdir(image_dir):
        if (file.find('.png') is not -1):
            src.append(image_dir + file)
            name.append(file)
            test.append(Dataization(image_dir + file))

    test = np.array(test)
    model = load_model('/content/COVIDMD.h5')
    predict = model.predict_classes(test)

    for i in range(len(test)):
        print(name[i] + " : , Predict : " + str(categories[predict[i]]))

@app.route("/predict", methods=["POST"])
def predict():
    # view로부터 반환될 데이터 딕셔너리를 초기화
    data = {"success": False}

    if request.method == "POST" and request.files.get("image"):
        image = request.files["image"].read()
        extention = magic.from_buffer(image).split()[0].upper()
        if extention == 'GIF':
            imageObject = Image.open(io.BytesIO(image))
            imageObject.seek(0)
            imageObject = imageObject.convert('RGB')
            image = np.array(imageObject)
        elif extention != 'JPEG' and extention != 'PNG':
            return jsonify(data)
        else:
            image = Image.open(io.BytesIO(image)).convert('RGB')
        image = prepare_image(image, target=(112, 112))
        # 입력 이미지를 분류하고 클라이언트로부터 반환되는 예측치들의 리스트를 초기화 합니다.
        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data["predictions"] = []

        # 결과를 반복하여 반환된 예측 목록에 추가
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)

        # 요청 성공
        data["success"] = True
        # JSON 형식으로 데이터 딕셔너리 반환
    return jsonify(data)




if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()
