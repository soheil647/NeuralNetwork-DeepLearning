import pandas as pd
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataPreProcess:
    def __init__(self, raw_data_path, process_data_path):
        self.raw_data_path = raw_data_path
        self.process_data_path = process_data_path
        self.features = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']

    def prepare_data(self):
        # load data
        def parse(x):
            return datetime.strptime(x, '%Y %m %d %H')
        dataset = pd.read_csv(self.raw_data_path, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
        dataset.drop('No', axis=1, inplace=True)

        # manually specify column names
        dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        dataset.index.name = 'date'

        # mark all NA values with 0
        dataset['pollution'].fillna(0, inplace=True)

        # drop the first 24 hours
        dataset = dataset[24:]

        # summarize first 5 rows
        # print(dataset.head(5))

        # save to file
        dataset.to_csv(self.process_data_path)

    def plot_data(self):
        # load dataset
        dataset = pd.read_csv(self.process_data_path, header=0, index_col=0)
        values = dataset.values

        # specify columns to plot
        groups = [0, 1, 2, 3, 5, 6, 7]
        i = 1

        # plot each column
        pyplot.figure()
        for group in groups:
            pyplot.subplot(len(groups), 1, i)
            pyplot.plot(values[:, group])
            pyplot.title(dataset.columns[group], y=0.5, loc='right')
            i += 1
        pyplot.show()

    def prepare_data_for_time_series(self):
        # convert series to supervised learning
        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [(self.features[j] + '(t-%d)' % i) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [(self.features[j] + '(t)') for j in range(n_vars)]
                else:
                    names += [(self.features[j] + '(t+%d)' % i) for j in range(n_vars)]

            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)

            return agg

        # load dataset

        dataset = pd.read_csv(self.process_data_path, header=0, index_col=0)
        values = dataset.values

        # integer encode direction
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])

        # ensure all data is float
        values = values.astype('float32')

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        # frame as supervised learning
        reframed = series_to_supervised(scaled, 1, 1)

        # drop columns we don't want to predict
        reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
        print(reframed.head(10))
        # print(reframed)

    def do_action(self):
        self.prepare_data()
        # self.plot_data()
        self.prepare_data_for_time_series()
