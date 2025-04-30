from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os
import uuid
import cv2
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "Project.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z", "UNKNOWN"
]


UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)
    return img

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

@app.route("/", methods=["GET", "POST"])
def home():
    word = ""  
    filenames = []  
    uploaded_files = [] 

    if request.method == "POST":
        if "files" not in request.files:
            return render_template("index.html", error="No files uploaded")

        files = request.files.getlist("files") 
        if not files or files[0].filename == "":
            return render_template("index.html", error="No selected files")

        for file in files:
            original_filename = secure_filename(file.filename) 
            unique_filename = str(uuid.uuid4()) + "_" + original_filename 
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            uploaded_files.append((original_filename, file_path))

        uploaded_files.sort(key=lambda x: natural_sort_key(x[0])) 

        for _, file_path in uploaded_files:
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_letter = class_names[predicted_class] if 0 <= predicted_class < len(class_names) else "?"
            word += predicted_letter
            filenames.append(os.path.basename(file_path))

        return render_template("index.html", filenames=filenames, predicted_word=word)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
