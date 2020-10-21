# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_dropzone import Dropzone
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import glob
import os

app = Flask(__name__)
dropzone = Dropzone(app)
# Initialisation des modèles de prédiction
model = load_model('src/best_model_Xception_DataAugmentation_120_breeds.hdf5')
class_label = pd.read_csv('src/class_labels.csv')
path_tmp = 'static/tmp/'

@app.route('/')
def home():
    for file in glob.glob(path_tmp + '*.jpg'):
        os.remove(file)
    return render_template('form.html')

@app.route('/uploads', methods=['POST'])
def upload():
    # Récupération du fichier dans la dropzone
    f = request.files.get('file')
    f.save(os.path.join(path_tmp, f.filename))
    return render_template('form.html')


@app.route("/dogs", methods=["GET"])
def predict():
    img_lst = []
    breed_lst = []

    for file in glob.glob(path_tmp + '*.jpg'):
        img_lst.append(file)

        image = load_img(file, target_size=(299, 299))
        input_array = np.reshape(img_to_array(image), (-1, 299, 299, 3)) / 255

        # Prédiction de la race
        output = model.predict(input_array)
        breed = class_label.at[output.argmax(axis=-1)[0], 'index']

        breed_lst.append(breed)

    breed_lst = ','.join(breed_lst)
    img_lst = ','.join(img_lst).replace('\\', '/').replace('dev/api', '')
    return render_template('form.html', img_lst=img_lst, breed_lst=breed_lst)


if __name__ == "__main__":
    app.run(debug=True)
