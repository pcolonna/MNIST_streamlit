import streamlit as st
import model
import plot
import streamlit as st
import time
import numpy as np



# last_rows = np.random.randn(1, 1)]

st.sidebar.title('Mnist')


options = {'v0': 'foo', 'v1': 'bar'}

st.sidebar.info("This is a demo application of MNIST and Streamlit")
sidebar_select = st.sidebar.selectbox(
    'Menu', options=list(options.keys()), key='menu_option')


mnist_model = model.create(784)
summary = model.summarize(mnist_model)

col_left, col_right = st.beta_columns(2)

with col_left:
    chart = st.line_chart(np.array([0.]))
    status_text = st.empty()
    status_text.text("0 % Complete")

    val_acc_text = st.empty()
    val_acc_text.text("Best Validation Accuracy: 0")

    progress_bar = st.progress(0)

with col_right:
    st.markdown("<h1> Model summary </h1>", unsafe_allow_html=True)
    st.text(summary)

batch_size = st.sidebar.slider('Batch Size', min_value=16, max_value=256, step=16)
epoch = st.sidebar.slider('Epoch', min_value=10, max_value=1000)



if st.button('Train model'):
    progress_bar.progress(0)
    model.run_experiment(epoch, batch_size, progress_bar, status_text, val_acc_text, chart)

