# import the Package
import os
import cv2
import numpy as np 
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from PIL import Image

class_name = ["airplane", "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]

# Create a function to load my saved model
@st.cache(allow_output_mutation=True)
def load_my_model():
    model = tf.keras.models.load_model("final_model.h5")
    return model

model = load_my_model()

# Create a title of web App
st.title("Image Classification with Cifar10 Dataset")
st.header("Please Upload images related to this things...")
st.text(class_name)

# create a file uploader and take a image as an jpg or png
file = st.file_uploader("Upload the image" , type=["jpg" , "png"])

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

if st.button("Predict"):
    image = Image.open(file)
    st.image(image , use_column_width=True)
    
    img = load_image("./images/"+file.name)
    predictions = np.argmax(model.predict(img), axis=-1)

    class_name = ["airplane", "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]
    print(class_name[np.argmax(predictions)])
    string = "Image mostly same as : - " + class_name[predictions[0]]
    st.success(string)