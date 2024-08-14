import cv2
import streamlit as st

def plot_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    st.bar_chart(hist.flatten())
