from tensorflow.keras.callbacks import Callback

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from datetime import datetime
from time import time

import pandas as pd


class CallbackPlot(Callback):

    # =============================================================================
    #     Plots the passed losses and metrics, updating them at each
    #     epoch, and saving them at the end
    # =============================================================================
    def custom_pause(self, interval):
        '''
        Custom pause function that allows matplotlib to update the plot at each end of epoch wihout the window popping up
        '''
        manager = plt._pylab_helpers.Gcf.get_active()
        if manager is not None:
            canvas = manager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
            # plt.show(block=False)
            canvas.start_event_loop(interval)
        else:
            time.sleep(interval)

    def __init__(self, plots_settings, title,
                 folder_path='Training Plots/Training',
                 share_x=False):
        # =============================================================================
        #         Initializes figure
        #         plots_settings: tuple containing dictionaries. Each dictionary corresponds to settings of a plot (window)
        #               'variables': another dictionary, containing variables to be monitored, eg {'loss': 'Training loss', 'val_loss': 'Validation loss'}
        #               'title': title of window
        #               'ylabel': label of y axes
        #               'last_epochs': last epochs to plot. if not present in dictionary, the whole log will be ploted
        #         title: title of figure
        #         folder_path: path of folder that will contain all the training information
        #         share_x: either to share the X axis or not
        # =============================================================================

        super().__init__()
        self.plots_settings = plots_settings
        self.plot_count = len(plots_settings)
        self.title = title
        self.share_x = share_x
        self.folder_path = folder_path

        # list that keeps track of all drawn lists, so they can be cleared at each new plot
        self.line_list = []

    def on_train_begin(self, logs={}):
        plt.ion()

        self.figure, self.windows = plt.subplots(self.plot_count, 1, figsize=[15, 4*self.plot_count], clear=True, num=self.title,
                                                 sharex=self.share_x, constrained_layout=True, squeeze=False)
        self.custom_pause(0.1)
        # dictionary that contains losses and metrics throughout training
        self.losses_and_metrics_dict = {}

        for plot_settings in self.plots_settings:
            for variable in plot_settings['variables'].keys():
                self.losses_and_metrics_dict[variable] = []

        for window, plot_settings in zip(self.windows, self.plots_settings):

            window = window[0]

            if ('last_epochs' not in plot_settings):

                window.set(xlabel='Epoch', ylabel=plot_settings['ylabel'])
                window.set_title(plot_settings['title'])

            else:
                last_epochs = plot_settings['last_epochs']
                window.set(xlabel='Last ' + str(last_epochs) + ' Epochs',
                           ylabel=plot_settings['ylabel'])
                window.set_title(
                    plot_settings['title'] + ' on last ' + str(last_epochs) + ' epochs')

    def on_epoch_end(self, epoch, logs={}):

        variables_list = self.losses_and_metrics_dict.keys()

        for variable in variables_list:
            self.losses_and_metrics_dict[variable].append(logs.get(variable))

        # calls the right figure to modify it
        plt.figure(self.title)

        if epoch > 1:
            for line in self.line_list:
                line.remove()
            self.line_list = []

        if epoch > 0:

            for window, plot_settings in zip(self.windows, self.plots_settings):

                window = window[0]

                # clears plot
                # window.clear()
                plt.pause(0.05)

                # checks if the whole data is to be ploted
                if ('last_epochs' not in plot_settings) or (epoch < plot_settings['last_epochs']):

                    # plots all the variables for this plot
                    for variable, legend in plot_settings['variables'].items():
                        self.line_list.append(
                            window.plot(self.losses_and_metrics_dict[variable], label=legend)[0])

                else:
                    last_epochs = plot_settings['last_epochs']
                    # plots all the variables for this plot
                    for variable, legend in plot_settings['variables'].items():
                        self.line_list.append(
                            window.plot(range(
                                epoch - last_epochs + 1, epoch + 1), self.losses_and_metrics_dict[variable][-last_epochs:], label=legend)[0])

                    window.set_xlim(left=epoch - last_epochs + 1, right=epoch)

                window.legend()
                self.custom_pause(0.1)

    def on_train_end(self, logs={}):

        # saves losses abd metrics plot
        plt.figure(self.title, clear=False)
        plt.savefig(self.folder_path + '/' + self.title + '.png')
        self.custom_pause(0.1)


class ExperimentalPlotCallback(Callback):

    def __init__(self, plots_settings, title,
                 folder_path='Training Plots/Training',
                 share_x=False):
        # =============================================================================
        #         Initializes figure
        #         plots_settings: tuple containing dictionaries. Each dictionary corresponds to settings of a plot (window)
        #               'variables': another dictionary, containing variables to be monitored, eg {'loss': 'Training loss', 'val_loss': 'Validation loss'}
        #               'title': title of window
        #               'ylabel': label of y axes
        #               'last_epochs': last epochs to plot. if not present in dictionary, the whole log will be ploted
        #         title: title of figure
        #         folder_path: path of folder that will contain all the training information
        #         share_x: either to share the X axis or not
        # =============================================================================

        super().__init__()
        self.plots_settings = plots_settings
        self.plot_count = len(plots_settings)
        self.title = title
        self.share_x = share_x
        self.folder_path = folder_path

    def on_train_begin(self, logs={}):
        plt.ion()

        self.figure, self.windows = plt.subplots(self.plot_count, 1, figsize=[15, 4*self.plot_count], clear=True, num=self.title,
                                                 sharex=self.share_x, constrained_layout=True, squeeze=False)

        # sets axis labels and titles for windows
        for window, plot_settings in zip(self.windows, self.plots_settings):

            window = window[0]

            if ('last_epochs' not in plot_settings):
                window.set(xlabel='Epoch', ylabel=plot_settings['ylabel'])
                window.set_title(plot_settings['title'])

            else:
                last_epochs = plot_settings['last_epochs']
                window.set(xlabel='Last ' + str(last_epochs) +
                           ' Epochs', ylabel=plot_settings['ylabel'])
                window.set_title(
                    plot_settings['title'] + ' on last ' + str(last_epochs) + ' epochs')

        self.custom_pause(0.1)
        # dictionary that contains losses and metrics throughout training
        self.losses_and_metrics_dict = {}

        for plot_settings in self.plots_settings:
            for variable in plot_settings['variables'].keys():
                self.losses_and_metrics_dict[variable] = []

    def on_epoch_end(self, epoch, logs={}):

        variables_list = self.losses_and_metrics_dict.keys()

        for variable in variables_list:
            self.losses_and_metrics_dict[variable].append(logs.get(variable))

        for line in self.line_list:
            line.remove()

        self.custom_pause(0.1)

        self.line_list = []

        # calls the right figure to modify it
        plt.figure(self.title)

        if epoch > 1:

            for window, plot_settings in zip(self.windows, self.plots_settings):

                window = window[0]
                # clears plot
                # window.clear()
                self.custom_pause(0.1)

                # checks if the whole data is to be ploted
                if ('last_epochs' not in plot_settings) or (epoch < plot_settings['last_epochs']):

                    # plots all the variables for this plot
                    for variable, legend in plot_settings['variables'].items():
                        self.line_list.append(window.plot(
                            self.losses_and_metrics_dict[variable], label=legend)[0])

                else:
                    last_epochs = plot_settings['last_epochs']
                    # plots all the variables for this plot
                    for variable, legend in plot_settings['variables'].items():
                        self.line_list.append(window.plot(range(
                            epoch - last_epochs + 1, epoch + 1), self.losses_and_metrics_dict[variable][-last_epochs:], label=legend)[0])

                    self.custom_pause(0.1)
                    window.set_xlim(left=epoch-last_epochs+1, right=epoch)

                window.legend()
                self.custom_pause(0.1)

    def on_train_end(self, logs={}):

        # saves losses abd metrics plot
        plt.figure(self.title, clear=False)
        plt.savefig(self.folder_path + '/' + self.title + '.png')
        self.custom_pause(0.1)


class CallbackSaveLogs(Callback):

    # =============================================================================
    # Saves the configuration file and a training log table
    # =============================================================================

    def __init__(self, folder_path='Training Plots/Training'):

        self.folder_path = folder_path

    def on_train_begin(self, logs={}):

        # dictionary that contains values for monitored variables
        self.logs = {}
        self.logs['elapsed time'] = []

        # gets timestamp at start of training
        self.timestamp_start = datetime.now()

    def on_epoch_end(self, epoch, logs={}):

        # updates the last value for losses and metrics at the end of each epoch
        for variable, value in logs.items():

            if variable not in self.logs:
                self.logs[variable] = []

            self.logs[variable].append(value)

        # computes the time spent so far until that epoch
        delta = datetime.now() - self.timestamp_start

        # extract hours, minutes and seconds
        hours, remainder = divmod(delta.total_seconds(), 60*60)
        minutes, remainder = divmod(remainder, 60)
        seconds, _ = divmod(remainder, 1)

        # appends elapsed time
        self.logs['elapsed time'].append('{:02}:{:02}:{:02}'.format(
            int(hours), int(minutes), int(seconds)))

    def on_train_end(self, logs={}):

        # creates a pandas series of training logs
        columns = [variable for variable in self.logs.keys()]
        # move the 'elapsed time' column to the end of list
        columns.append(columns.pop(columns.index('elapsed time')))

        log_df = pd.DataFrame(self.logs, columns=columns, index=[
                              i for i in range(1, len(self.logs['elapsed time']) + 1)])

        log_df.index.name = 'epoch'

        # saves the training data to a csv file
        log_df.to_csv(self.folder_path + '/' +
                      'Training logs.csv')
