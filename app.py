import pytesseract as pt
import streamlit as st
import cv2
import numpy as np
from pytesseract import Output
import matplotlib.pyplot as plt

st.sidebar.markdown("""<style>body {background-color: #2C3454;color:white;}</style><body></body>""", unsafe_allow_html=True)
st.markdown("""<h1 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>CROWDSHAKTI</h1><h1 style='text-align: center; color: white;font-size:30px;margin-top:-30px;'>Machine Learning <br></h1>""",unsafe_allow_html=True)

image_file = st.sidebar.file_uploader("", type = ["jpg","png","jpeg"])

def extract(img):
    slide=st.sidebar.slider("Select Page Segmentation Mode (Oem)",1,4)
    slide=st.sidebar.slider("Select Page Segmentation Mode (Psm)",1,14)
    conf=f"-l eng --oem 3  --psm {slide}"
    text = pt.image_to_string(img, config=conf)
    st.markdown("<h1 style = 'color:yellow;'>Extracted Text</h1>", unsafe_allow_html = True)
    if text != "":
      slot = st.empty()
      slot.markdown(f"{text}")
      
      
    d = pt.image_to_data(img,output_type = Output.DICT)
    st.markdown("<h1 style = 'color:yellow;'>Extracted Image</h1>", unsafe_allow_html = True)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if(d['text'][i] != ""):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    plt.imshow(img)
    st.image(img, use_column_width = True, clamp = True)
     
if image_file is not None:
    st.markdown("<h1 style = 'color:yellow;'>Uploaded Image</h1>", unsafe_allow_html = True)
    st.image(image_file, width = 400)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    radio=st.sidebar.radio("Select Action",('Oem','Psm'))
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if(radio=="Oem"):
        
        extract(img)
    else:
        (radio=="psm")
        extract(img)
