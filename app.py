import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.title('OCR App with EasyOCR')
st.sidebar.info('Created by Lintang Gilang')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[1]
    if file_type not in ["jpg", "jpeg", "png"]:
        st.warning('Please upload a valid image format (jpg, jpeg, png) and try again.')
    else:
        col1, col2 = st.columns(2)  # Updated from beta_columns
        
        # Convert uploaded BytesIO stream to OpenCV format
        uploaded_file_bound = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(uploaded_file_bound, 1)  # Convert byte to matrix & 1 means read the image in color
        
        # Initialize OCR reader
        reader = easyocr.Reader(['en'], gpu=False)
        bounds = reader.readtext(img, detail=1, paragraph=False)
        
        # Drawing bounding boxes on the image
        for (coord, text, prob) in bounds:
            (topleft, topright, bottomright, bottomleft) = coord
            tx, ty = (int(topleft[0]), int(topleft[1]))
            bx, by = (int(bottomright[0]), int(bottomright[1]))
            cv2.rectangle(img, (tx, ty), (bx, by), (0, 0, 255), 2)
        
        # Convert the OpenCV image to PIL format for Streamlit
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Display the uploaded image with bounding boxes in the left column
        with col1:
            st.write("Detection...")
            st.image(img_pil, caption='Uploaded Image with Bounding Boxes.', use_column_width=True)
        
        # Display the OCR result in the right column
        with col2:
            st.write("Recognizing...")
            
            # Extracting text results
            results = [text for coord, text, prob in bounds]

            df = pd.DataFrame()
            df['result'] = results

            st.table(df)
