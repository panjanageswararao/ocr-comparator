import streamlit as st
from PIL import Image
import pytesseract as pt
import cv2 as cv
import cv2
import pytesseract
import matplotlib.pyplot as plt 
from pytesseract import Output
import numpy as np
import easyocr
reader = easyocr.Reader(['en'])
#pt.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'c:\Users\Panja\AppData\Local\Programs\Tesseract-OCR\\tesseract.exe'

st.sidebar.markdown("""<style>body {background-color: #2C3454;color:white;}</style><body></body>""", unsafe_allow_html=True)
st.markdown("""<h1 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>CROWDSHAKTI</h1><h1 style='text-align: center; color: white;font-size:30px;margin-top:-30px;'>Machine Learning <br></h1>""",unsafe_allow_html=True)

def intro():
    st.markdown("""<h2 style='text-align: left; color: white;'>Problem Statement</h2><p style='color: white;'>Love knows no gender and the LGBTQ (Lesbian, Gay, Bisexual, Transgender, and Queer) community is the epitome of this thought. <br>In honor of Pride Month, we are here with another Machine Learning challenge, in association with Pride Circle, to celebrate the impact and changes that they made globally.<br>You have been appointed as a social media moderator for your firm. <br>Your key responsibility is to tag and categorize quotes that are uploaded during Pride Month on the basis of its sentiment—positive, negative, and random. <br>Your task is to build a sophisticated Machine Learning model combining Optical Character Recognition (OCR) and Natural Language Processing (NLP) to assess sentiments of these quotes.</p>""",unsafe_allow_html=True)
    st.markdown("""<h2 style='text-align: left; color: white;'>TASK</h2><p style='color: white;'>You need to perform OCR on Images to extract text and then perform sentiment analysis on the extracted texts and classify them into positive, negative, or random.</p>""",unsafe_allow_html=True)
    st.markdown("""<h2 style='text-align: left; color: white;'>My Approach</h2><p style='color: white;'>
    <ul style='text-align: left; color: white;'><li>Extract Text from Images using Pytesseract API</li>
    <li>Get Sentiment Polarity of extracted text using TextBlob</li>
    <li>Classify texts as:<br>Positive if polarity is greater than 0,<br>Negative if polarity is less than 0,<br>Random if polarity is 0 or length of extracted text is 0</li></ul></p>""",unsafe_allow_html=True)
    st.markdown("""<a style='text-align: center; color: white;font-size:30px;' href="https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-pride-month-edition/leaderboard/detect-the-sentiment-of-a-quote-2-ca749be7/page/3/" target="_blank">LeaderBoard</a>""",unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center;color: #2C3454;margin-top:30px;margin-bottom:-20px;'>Select Image</h1>", unsafe_allow_html=True)

image_file = st.sidebar.file_uploader("", type=["jpg","png","jpeg"])



def extract(img):
    res=reader.readtext(img)
    for (bbox, text, prob) in res:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        
        cv2.rectangle(img, tl, br, (0, 255, 0), 5)
            
if image_file is not None:
    st.markdown("<h1 style='color:yellow;'>Uploaded Image</h1>", unsafe_allow_html=True)
    st.image(image_file,width=400)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    radio=st.sidebar.radio("Select Action",('EasyOCR','Pytesseract'))
    img = cv.imdecode( file_bytes, cv.IMREAD_COLOR)
    st.markdown("<h1 style='color:yellow;'>Extracted Image</h1>", unsafe_allow_html=True)
    if (radio =='EasyOCR'):
        extract(img)
        st.image(img, use_column_width=True,clamp = True)
       
    else:
        
        d = pytesseract.image_to_data(img,output_type=Output.DICT)
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if(d['text'][i] != ""):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.imshow(img)
        st.image(img, use_column_width=True,clamp = True)
        

   
  


  
            
    

   



