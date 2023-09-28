import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 載入訓練好的模型
model = load_model("dog_detector_model.keras")

# 定義一個函數來預測圖片中是否有小狗
def predict_dog(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 將圖片轉換成模型所需的形狀
    img_array /= 255.0  # 正規化圖片像素值
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        return "這張圖片中有小狗"
    else:
        return "這張圖片中沒有小狗"

# 替換下面的路徑為你想要進行預測的圖片路徑
image_path = "/Users/jackchen/Desktop/913686.jpg"
result = predict_dog(image_path)
print(result)