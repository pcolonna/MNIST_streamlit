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

batch_size = st.sidebar.slider('Batch Size', min_value=16, max_value=256, step=16)
epoch = st.sidebar.slider('Epoch', min_value=10, max_value=1000)


st.title("Training")

st.markdown("**You can launch a model and watch its performance improves. **")


mnist_model = model.create(784)

col_left, col_right = st.beta_columns(2)

with col_left:
    st.markdown("## Performance")
    chart = st.line_chart(np.array([0.]))
    status_text = st.empty()
    status_text.markdown("** Completed: **  0 %")

    val_acc_text = st.empty()
    val_acc_text.markdown("**Best Validation Accuracy:** 0")

    progress_bar = st.progress(0)

    if st.button('Train model'):
        progress_bar.progress(0)
        model.run_experiment(epoch, batch_size, progress_bar, status_text, val_acc_text, chart)

with col_right:
    st.markdown("<h2> Model summary </h2>", unsafe_allow_html=True)
    summary = model.summarize(mnist_model)
    st.text(summary)

st.title("Inference")

st.markdown("**You can pick a number at random and predict with your trained model.**")

left, right =st.beta_columns(2)

with right:
    ground_truth_text = st.empty()
    prediction_text = st.empty()

with left:
    image = np.zeros((28,28))
    canvas = st.image(image, width=150)
    
    if st.button('Predict'):
        
        missing_model_text = st.empty()
        model.predict(missing_model_text, canvas, ground_truth_text, prediction_text)

