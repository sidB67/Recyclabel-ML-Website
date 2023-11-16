import streamlit as st
from PIL import Image, ImageOps
from ml import predict
st.title("ECOBIN")
st.header("Recyclable Waste Classification Example")
uploaded_file = st.file_uploader("Upload waste material image", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width = 200, caption='Uploaded MRI.')
    st.write("Classifying...")
    pred = predict(image)
    if pred <= 0.5:
        st.write("The waste is Non-Recyclable")
    else:
        st.write("The waste is Recyclable")
else :
    st.write("Please upload an image to classify")
