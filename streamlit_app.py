import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import os


# Chargement du model
saved_w = "premiers_poids.h5"
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(saved_w)

def predict_old(upload):

    img = Image.open(upload)
    img = np.asarray(img)
    img_resize = cv2.resize(img, (224, 224))
    img_resize = np.expand_dims(img_resize, axis=0)
    pred = model.predict(img_resize)

    rec = pred[0][0]

    return rec




st.title("Reconnaissance d'animaux sauvages")

upload = st.file_uploader("Chargez l'image que vous souhaitez détecter",
                           type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
    
    img = Image.open(BytesIO(upload.getvalue()))
    
    img = img.resize((224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    #traitement de l'image
    #img_processed = preprocess(img)

    # predictions
    rec = model.predict(img)
    
    predicted_classes = np.argmax(rec, axis=1)
    if 0.1 <= rec[0][0] <= 0.9:
        c2.write("No Wild boar or Deer detected on this picture")
    elif predicted_classes == 0: 
        c2.write("It is a wild boar. Il s'agit d'un sanglier (wild boar)")
    else:
        c2.write("It is a deer. Il s'agit d'un cerf (deer)")
    c1.image(Image.open(upload))
    
