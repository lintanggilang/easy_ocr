import streamlit as st
import easyocr
import pandas as pd

st.title('OCR App with EasyOCR')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.beta_columns(2)
    
    # Display the uploaded image in the left column
    with col1:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # OCR process and display the result in the right column
    with col2:
        st.write("Recognizing...")
        
        # Convert the uploaded BytesIO stream to bytes
        uploaded_file_bytes = uploaded_file.getvalue()

        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(uploaded_file_bytes, detail=0, paragraph=True)

        df = pd.DataFrame()
        df['result'] = result

        st.table(df)
