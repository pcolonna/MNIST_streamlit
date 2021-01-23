# updatable plot
# a minimal example (sort of)
import tensorflow as tf
from tensorflow import keras
from bokeh.plotting import figure
import streamlit as st

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.p = figure(
            title='simple line example',
            x_axis_label='x',
            y_axis_label='y')
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        # plt.plot(self.x, self.losses, label="loss")
        # plt.plot(self.x, self.val_losses, label="val_loss")
        # plt.legend()
        self.p.line(self.x, self.val_losses, line_width=2)
        st.bokeh_chart(self.p, use_container_width=True)