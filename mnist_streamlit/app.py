import streamlit as st
import model
import plot
import streamlit as st
import time
import numpy as np

import pprint
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import cv2


# last_rows = np.random.randn(1, 1)]

st.sidebar.title("Mnist")


options = {"v0": "foo", "v1": "bar"}

st.sidebar.info("This is a demo application of MNIST and Streamlit")
sidebar_select = st.sidebar.selectbox(
    "Menu", options=list(options.keys()), key="menu_option"
)

batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, step=16)
epoch = st.sidebar.slider("Epoch", min_value=10, max_value=1000)


st.title("Training")

st.markdown("**You can launch a model and watch its performance improves.**")


mnist_model = model.create(784)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("## Performance")
    chart = st.line_chart(np.array([0.0]))
    status_text = st.empty()
    status_text.markdown("** Completed: **  0 %")

    val_acc_text = st.empty()
    val_acc_text.markdown("**Best Validation Accuracy:** 0")

    progress_bar = st.progress(0)

    if st.button("Train model"):
        progress_bar.progress(0)
        model.run_experiment(
            epoch, batch_size, progress_bar, status_text, val_acc_text, chart
        )

with col_right:
    st.markdown("<h2> Model summary </h2>", unsafe_allow_html=True)
    summary = model.summarize(mnist_model)
    st.text(summary)

st.title("Inference")

st.markdown("**You can pick a number at random and predict with your trained model.**")

left, right = st.columns(2)

with right:
    ground_truth_text = st.empty()
    prediction_text = st.empty()

with left:
    image = np.zeros((28, 28))
    canvas = st.image(image, width=150)

    if st.button("Predict"):
        missing_model_text = ""  # st.empty()
        model.predict(canvas, ground_truth_text, prediction_text, missing_model_text)


st.markdown("**You can also draw a number and see the prediction.**")


left, right = st.columns(2)

with right:
    drawing_prediction_text = st.empty()

with left:
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="" if bg_image else bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=150,
        width=150,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Do something interesting with the image data and paths
    # if canvas_result.image_data is not None:
    # st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        # st.text(canvas_result)

        # grayscale_image = 255 - canvas_result.image_data[:,:,3]
        grayscale_image = np.dot(
            canvas_result.image_data[..., :3], [0.299, 0.587, 0.114]
        )
        # st.text(255 - canvas_result.image_data[:,:,3])
        # pprint.pprint(list(grayscale_image[0]))
        # st.text(grayscale_image)
        # st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))

        resized_image = cv2.resize(
            grayscale_image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC
        )
        # st.text(list(grayscale_image))

    if st.button("Predict from drawing"):
        missing_model_text_drawing = st.empty()
        model.predict_from_drawing(
            resized_image, drawing_prediction_text, missing_model_text_drawing
        )
