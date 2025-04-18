import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle  #to load a saved model
import base64  #to open .gif files in streamlit app
from tensorflow.keras.preprocessing import image
import kagglehub
import time
from io import StringIO



# @st.cache_data(suppress_st_warning=True)
@st.cache_data
def get_fvalue(val):    
	feature_dict = {"No":1,"Yes":2}    
	for key,value in feature_dict.items():        
		if val == key:            
			return value
def get_value(val,my_dict):    
	for key,value in my_dict.items():        
		if val == key:            
			return value

app_mode = st.sidebar.selectbox(':green[Select Page]',['Home','Upload and Predict']) #two pages

css="""
<style>
    [data-testid="stSidebar"] {
        background: #FFFFFF;
		# color: blue;
    }
</style>
"""
st.write(css, unsafe_allow_html=True)


# Download latest version
path = kagglehub.model_download("utkarshsaxenadn/ai-vs-human/tensorFlow2/default")
print("Path to model files:", path)

# Load the trained model
path = path+'/ResNet50V2-AIvsHumanGenImages.keras'
model = load_model(path)

if app_mode=='Home':   
    st.title('AI versus Human Intelligence')
    st.write("Upload images to determine if they are AI or Human generated images")
    st.image('ai-human.jpg')    

elif app_mode == 'Upload and Predict':     
	st.subheader('Is it an AI or Human generated image ?')    
	img_path = st.file_uploader("Please upload an image")
	time.sleep(1)
	if st.button("Predict"):        
		img = image.load_img(img_path, target_size=(512, 512))  # ResNet50V2 input size
		img_array = image.img_to_array(img) / 255.0  # Normalize
		img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch processing


		# file_ = open("einstein.jpg", "rb")        
		# contents = file_.read()        
		# data_url = base64.b64encode(contents).decode("utf-8")        
		# file_.close()        

		# file = open("ai.jpg", "rb")        
		# contents = file.read()        
		# data_url_no = base64.b64encode(contents).decode("utf-8")
		# file.close()     
		
                # Predict
		prediction = model.predict(img_array)
		
		if prediction[0] > 0.5 :            
			st.success('This is an AI generated image')
			st.image(img_path, caption='AI generated image', use_container_width=True)
		elif prediction[0] <= .5 :
			st.success('This is a Human generated image')
			st.image(img_path, caption='Human generated image', use_container_width=True)
	
