import sys
import os
import glob
import re
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Load your trained model
MODEL = load_model("saved_model")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256, 3))

    # Preprocessing the image
    img = image.img_to_array(img)
    img = img / 255
    img_batch = np.expand_dims(img, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class+' with '+'confidence: '+str(confidence)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, MODEL)

        dir = './uploads'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

        return result
    return None


if __name__ == '__main__':
    app.run(port=5001, debug=True)
