import numpy as np


class TimeSeriesOutlier:
    def __init__(self, dims=3, delay=1, offset=0.0, wrap=True, **kwargs):
        '''
        Superclass providing standard access methods for LOF and TOF

        :param int dims:            (Optional) Embedding dimension (default 3)
        :param int delay:           (Optional) Number of indices in the data to offset for time delay (default 1)
        :param float offset:        (Optional) Relative position within window to map sample to embedded coordinate (default 0)
        :param bool wrap:           (Optional) Whether or not to wrap data to preserve number of samples (default true)
        '''
        self.dims = dims
        self.delay = delay
        self.offset = offset
        self.wrap = wrap
        self.unused_kwargs = kwargs

    
    def _time_delay_embed(self, data):
        '''
        Generate and save an array representing the time-embedded phase space of an input time series

        :param numpy.ndarray data: 1D time series to be embedded
        '''
        width = int(self.dims*self.delay)
        data_length = data.shape[0]
        if self.wrap:
            offset_idx = int(self.offset*width)
            end_padding = width - offset_idx - 1
            if offset_idx > 0:
                data = np.insert(data, 0, data[-offset_idx:])
            if end_padding > 0:
                data = np.append(data, data[0:end_padding])
        else:
            data_length -= (width-1)
        
        indexer_row = np.arange(0, width, self.delay)
        indexer_col = np.arange(data_length).reshape(-1, 1)
        indexer = np.tile(indexer_row, (data_length, 1)) + np.tile(indexer_col, (1, self.dims))
        self.embedded_data = data[indexer]

    
    def _set_embedded_data(self, data):
        self.embedded_data = data


    def _set_truncated_data(self, data, times, length):
        self.trunc_data = data[:length]
        if times is not None:
            self.trunc_times = times[:length]
        else:
            self.trunc_times = np.arange(0, length)
    

    def get_embedded_data(self):
        '''
        Get array of embedded data

        :return: data array of shape (n_samples, n_dimensions)
        :rtype: numpy.ndarray
        '''
        return self.embedded_data


    def get_truncated_data(self):
        '''
        Get data and time labels corresponding to all processed samples

        :return: data array, time label array
        :rtype: (numpy.ndarray, numpy.ndarray)
        '''
        return self.trunc_data, self.trunc_times

    def get_outlier_indices(self):
        '''
        Get a list of data indices corresponding to detected outliers
        :return: array of indices
        :rtype: numpy.ndarray
        '''
        pass

    def get_outlier_factors(self):
        '''
        Get a list of all the labelled outlier factors for the fitted time series

        :return: array of floats
        :rtype: numpy.ndarray
        '''
        pass

    def get_neighbor_indices(self):
        '''
        :return: (samples, n_neighbors) matrix of the n_neighbors nearest neighbors of each sample
        :rtype: numpy.ndarray
        '''
        pass
