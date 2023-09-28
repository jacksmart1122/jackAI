from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO  # 新增這一行

app = Flask(__name__, template_folder='/Users/jackchen/Documents/FrontEnd-jack/')

# 載入已訓練好的模型
model = tf.keras.models.load_model('dog_detector_model.keras')

@app.route('/')
def index():
    return render_template('111.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image provided."

    img_file = request.files['image']
    if img_file.filename == '':
        return "No selected image file."

    # 讀取圖片並轉換為 BytesIO
    img_bytes = BytesIO()
    img_file.save(img_bytes)
    img_bytes.seek(0)  # 將游標重置為開始位置

    img = image.load_img(img_bytes, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "這張圖片中" + ("有小狗" if prediction[0][0] > 0.5 else "沒有小狗")

    # 返回預測結果
    return result

if __name__ == "__main__":
    app.run()








繼續生成
