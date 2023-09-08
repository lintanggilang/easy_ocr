import streamlit as st
import easyocr
import pandas as pd

st.title('OCR App with EasyOCR')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Recognizing...")

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(uploaded_file, detail=0, paragraph=True)

    df = pd.DataFrame()
    df['result'] = result

    st.dataframe(df)
