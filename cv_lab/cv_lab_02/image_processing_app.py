import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from processing import process_image, get_processing_functions
from utils import plot_histogram

# Streamlit app layout
st.title("Image Processing for Computer Vision Lab")
st.markdown("""---""")  # Adds a horizontal line below the heading

st.write("Upload an image and choose parameters to process the image.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Select processing function
    processing_functions = get_processing_functions()
    function_name = st.selectbox("Choose a processing function", list(processing_functions.keys()))

    # Parameter input
    params = {}

    if function_name == 'Power Law Transform':
        gamma_input = st.text_input("Input Value for gamma", value="2.0")
        try:
            params['gamma'] = float(gamma_input)  # Convert to float
        except ValueError:
            st.error("Please enter a valid number for gamma.")
            params['gamma'] = 2.0  # Default value or handle as appropriate

    if function_name == 'Contrast Stretching':
        params['r1'] = st.slider("r1 for Contrast stretching", 0, 255, 0)
        params['s1'] = st.slider("s1 for Contrast stretching", 0, 255, 0)
        params['r2'] = st.slider("r2 for Contrast stretching", params['r1'], 255, 0)
        params['s2'] = st.slider("s2 for Contrast stretching", params['s1'], 255, 0)

    if function_name == 'Gray Level Slicing':
        params['low_limit'] = st.slider("Low Limit for Gray Level Slicing", 0, 255, 100)
        params['high_limit'] = st.slider("High Limit for Gray Level Slicing", params['low_limit'], 255, 150)

    if function_name == 'Bright Spot Detection':
        params['low_limit'] = st.slider("Threshold for Bright Spot Detection", 0, 255, 200)

    st.markdown("""---""")  # Adds a horizontal line after the input parameters

    # Process image
    processed_image = process_image(image, function_name, **params)

    # Display options for original image
    display_option = st.radio("Display Original Image in:", ("Black and White","Color"))

    if display_option == "Black and White":
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = "GRAY"
    else:
        display_image = image
        channels = "BGR"

    # Display original and processed images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(display_image, caption='Original Image', use_column_width=True, channels=channels)

    with col2:
        st.image(processed_image, caption=f'{function_name}', use_column_width=True)

    # Display histograms
    st.write("Histogram of Original Image")
    plot_histogram(cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY) if channels == "BGR" else display_image)

    st.write("Histogram of Processed Image")
    plot_histogram(processed_image)
