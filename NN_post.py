import tensorflow.keras.applications.efficientnet as eff_net
import tensorflow as tf
import numpy as np
import PIL
import io
import base64
import cv2 as cv

from flask import Flask, jsonify, request

IMAGE_SIZE = (128, 128, 3)
map_classes = {0: "airplane", 1: "bicycle", 2: "boat", 3: "bus",
               4: "car", 5: "motorcycle", 6: "train", 7: "truck"}

def preprocess_input_model(_image):
    _image = _image.replace("data:image/jpeg;base64,", "")
    _image = PIL.Image.open(io.BytesIO(base64.b64decode(_image)))
    
    crop_image = _image.crop((20, 30, 110, 90))
    crop_image = crop_image.resize((128, 128), PIL.Image.ANTIALIAS)
    
    crop_image = cv.cvtColor(np.array(crop_image), cv.COLOR_BGRA2BGR)
    
    return eff_net.preprocess_input(crop_image)[None, ...]

def get_compiled_model():  
    base_model = eff_net.EfficientNetB7(include_top=False,
                                        weights="imagenet",
                                        input_shape=(IMAGE_SIZE))
    base_model.trainable = False

    model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(8, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.3e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    
    return model

conv_NN = get_compiled_model()
conv_NN.load_weights(r"efficientnetb7/best_model.hdf5")

app = Flask(__name__)
client = app.test_client()

data = [
    {
        "image": None,
        "answer": None
    }
]

@app.route("/predict", methods=["GET"])
def get_list():
    return jsonify(data)

@app.route("/predict", methods=["POST"])
def update_list():
    new_one = request.json
    
    base64_image = new_one["image"]
    
    image = preprocess_input_model(base64_image)
    pred = np.argmax(conv_NN.predict(image), axis=-1)
    
    answer = [
        {
            "image": base64_image,
            "answer": map_classes[pred[0]]
        }
    ]
    
    return jsonify(answer)

if __name__ == "__main__":
    app.run()
