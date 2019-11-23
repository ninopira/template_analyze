import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from keras.callbacks import EarlyStopping, LambdaCallback, CSVLogger
from keras.layers import Input, Dense, Dropout
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import adam
import pandas as pd


class Mlp():
    def __init__(self, input_size, result_dir, num_class=2, dropout_rate=0.1, learning_rate=0.01, loss=binary_crossentropy):
        """

        Parameters
        ----------
        input_size : tuple, (int, int)
        num_class : int
            number of class.
        kernel_size : tuple, (int, int)
            size of conv filter.
        dropout_rate : float
            If dropout_rate!=None, employ dropout.
        optimizer : keras.optimizers.*
        loss: keras.losses.*
        """
        self.input_size = input_size
        self.result_dir = result_dir
        self.num_class = num_class
        self.dropout_rate = dropout_rate
        self.optimizer = adam(decay=learning_rate)
        self.csv_path = os.path.join(self.result_dir, 'history.csv')
        self.png_path = os.path.join(self.result_dir, 'history.png')
        self.model_path = os.path.join(self.result_dir, 'model.h5')
        self.callbacks = self.build_callbacks()
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.input_size, ), name='input')
        dense_1_layer = Dense(32, activation='relu', name='Dence_1')(input_layer)
        dropout_1 = Dropout(self.dropout_rate)(dense_1_layer)
        if self.num_class == 2:
            output_layer = Dense(1, activation='sigmoid')(dropout_1)
        else:
            output_layer = Dense(self.num_class, activation='softmax')(dropout_1)
        model = Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=self.optimizer,
            loss=binary_crossentropy,
            metrics=['crossentropy']
        )
        print(model.summary())
        return model

    def build_callbacks(self):
        callbacks = [
            EarlyStopping(monitor='val_loss', verbose=1, patience=5, mode='auto'),
            CSVLogger(self.csv_path),
            self.build_plot_learnig_curve_callback()
        ]
        return callbacks

    def build_plot_learnig_curve_callback(self):
        def on_epoch_end(epoch, _):
            if (epoch+1) % 5 == 0 or epoch == 0:
                plot_learning_curve()

        def plot_learning_curve():
            df = pd.read_csv(self.csv_path)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(df['epoch'], df['loss'], label='train')
            ax.plot(df['epoch'], df['val_loss'], label='val')
            ax.get_xaxis().get_major_formatter().set_useOffset(False)
            ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.legend()
            fig.savefig(self.png_path)
        plot_learnig_curve_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        return plot_learnig_curve_callback

    def fit(self, x, y, batch_size, epochs, validation_data):
        history = self.model.fit(x, y, batch_size, epochs, validation_data=validation_data, callbacks=self.callbacks, verbose=1)
        return history

    def save(self):
        self.model.save(self.model_path, overwrite=True)






