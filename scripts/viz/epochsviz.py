#!/usr/bin/python3
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from functools import partial
from threading import Thread
from tornado import gen
import pickle



class Epochsviz:
    """docstring for ClassName"""

    def __init__(self, title='figure', plot_width=600, plot_height=600,
                 name_train_curve='Training loss', color_train='red',
                 name_val_curve='Validation loss', color_val='green',
                 line_width_train=2, line_width_val=2):

        self.source = ColumnDataSource(data={'epochs': [],
                                             'trainlosses': [],
                                             'vallosses': [] }
        )
        self.title = title
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.name_train_curve = name_train_curve
        self.name_val_curve = name_val_curve
        self.color_train = color_train
        self.color_val = color_val
        self.line_width_train=line_width_train
        self.line_width_val=line_width_val

        self.plot = figure(plot_width=self.plot_width,
                           plot_height=self.plot_height,
                           title=self.title)

        self.plot.line(x= 'epochs', y='trainlosses',
                       color=self.color_train, legend=self.name_train_curve,
                       line_width=self.line_width_train, alpha=0.8,
                       source=self.source)

        self.plot.line(x= 'epochs', y='vallosses',
                       color=self.color_val, legend=self.name_val_curve,
                       line_width=self.line_width_val, alpha=0.8,
                       source=self.source)
        self.doc = curdoc()
        self.doc.add_root(self.plot)


        with open("color_list.pkl", "rb") as f:
            self.list_color = pickle.load(f)

        assert (self.color_train in self.list_color),(
            f"The color {self.color_train} is not in the list_color of available"
            "colors. "
            "To see the list of is_available color, call Epochsviz().list_color"
            )

        assert (self.color_val in self.list_color),(
            f"The color {self.color_val} is not in the list_color of available"
            "colors. "
            "To see the list of is_available color, call Epochsviz().list_color"
            )


    @gen.coroutine
    def update(self, new_data):
        self.source.stream(new_data)


    def send_data(self, current_epoch, current_train_loss, current_val_loss):
        new_data = {'epochs': [current_epoch],
                    'trainlosses': [current_train_loss],
                    'vallosses': [current_val_loss] }

        self.doc.add_next_tick_callback(partial(self.update, new_data))

    def start_thread(self, train_function):
        thread = Thread(target=train_function)
        thread.start()



