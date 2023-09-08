import streamlit as st
import easyocr
import pandas as pd
import cv2
import numpy as np
from PIL import Image

st.title('OCR App with EasyOCR')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.beta_columns(2)
    
    # Convert uploaded BytesIO stream to OpenCV format
    uploaded_file_bound = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(uploaded_file_bound, 1)  # Convert byte to matrix & 1 means read the image in color
    
    # OCR process
    reader = easyocr.Reader(['en'], gpu=False)
    bound = reader.readtext(img, detail=1, paragraph=False)
    
    for (coord, text, prob) in bound:
        (topleft, topright, bottomright, bottomleft) = coord
        tx, ty = (int(topleft[0]), int(topleft[1]))
        bx, by = (int(bottomright[0]), int(bottomright[1]))
        cv2.rectangle(img, (tx, ty), (bx, by), (0, 0, 255), 2)
    
    # Convert the OpenCV image to PIL format for Streamlit
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Display the uploaded image with bounding boxes in the left column
    with col1:
        st.image(img_pil, caption='Uploaded Image with Bounding Boxes.', use_column_width=True)
    
    # Display the OCR result in the right column
    with col2:
        st.write("Recognizing...")
        
        # Convert the uploaded BytesIO stream to bytes
        uploaded_file_bytes = uploaded_file.getvalue()

        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(uploaded_file_bytes, detail=0, paragraph=True)

        df = pd.DataFrame()
        df['result'] = result

        st.table(df)
