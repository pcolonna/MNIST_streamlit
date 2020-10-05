import streamlit as st
import model

st.sidebar.title('Mnist')


options = {'v0': 'foo', 'v1': 'bar'}

st.sidebar.info("This is a demo application of MNIST and Streamlit")
sidebar_select = st.sidebar.selectbox(
    'Menu', options=list(options.keys()), key='menu_option')


mnist_model = model.create(784)
summary = model.summarize(mnist_model)

st.markdown("<h1> Model summary </h1>", unsafe_allow_html=True)
st.text(summary)

batch_size = st.sidebar.slider('Batch Size', min_value=16, max_value=256, step=16)
epoch = st.sidebar.slider('Epoch', min_value=10, max_value=1000)

if st.button('Train model'):
	model.run_experiment(epoch, batch_size)
