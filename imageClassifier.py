import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle  #to load a saved model
import base64  #to open .gif files in streamlit app
from tensorflow.keras.preprocessing import image
import kagglehub

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
app_mode = st.sidebar.selectbox(':blue[Select Page]',['Home','Prediction']) #two pages

css="""
<style>
    [data-testid="stSidebar"] {
        background: LightBlue;
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
    st.title('AI versus Human Image Classification:')      
    st.image('ai-human.jpg')    

elif app_mode == 'Prediction':     
	st.subheader('Please upload a photo !')    
	
	if st.button("Predict"):        
		file_ = open("einstein.jpg", "rb")        
		contents = file_.read()        
		data_url = base64.b64encode(contents).decode("utf-8")        
		file_.close()        

		file = open("ai.jpg", "rb")        
		contents = file.read()        
		data_url_no = base64.b64encode(contents).decode("utf-8")
		file.close()     

		img_path = 'stallone.jpeg'  # Replace with actual image path
		img = image.load_img(img_path, target_size=(512, 512))  # ResNet50V2 input size
		img_array = image.img_to_array(img) / 255.0  # Normalize
		img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch processing
		
                # Predict
		prediction = model.predict(img_array)
		
		if prediction[0] < 0.5 :            
			st.error('This is an AI generated image')
			st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">', unsafe_allow_html=True,)
		elif prediction[0] >= .5 :
			st.success('This is a Human generated image')
			st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">', unsafe_allow_html=True,)
		else:
			st.success('Not Sure!')
