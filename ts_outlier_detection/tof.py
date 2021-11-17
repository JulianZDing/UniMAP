import numpy as np
from sklearn.neighbors import NearestNeighbors

from ts_outlier_detection.time_series_outlier import TimeSeriesOutlier

class TemporalOutlierFactor(TimeSeriesOutlier):
    def __init__(self, n_neighbors=None, event_length=80, q=2, **kwargs):
        '''
        Detects unique events ("unicorns") in one-dimensional time series data

        :param int n_neighbors:  (Optional) Number of nearest neighbors to consider in outlier factor calculation (default dims+1)
        :param int event_length: (Optional) Maximum detectable event length (samples); sets TOF detection threshold (default 80)
        :param int q:            (Optional) Exponent degree to use in TOF calculation (default 2)

        Remaining parameters are passed to sklearn.neighbors.NearestNeighbors
        '''
        super().__init__(**kwargs)
        n_neighbors = self.dims+1 if n_neighbors is None else n_neighbors
        self.kNN = NearestNeighbors(n_neighbors=n_neighbors, **self.unused_kwargs)
        self.n_neighbors = n_neighbors
        self._set_threshold(event_length)
        self.event_length = event_length
        self.q = q
    

    def _set_threshold(self, event_length):
        '''
        Set TOF threshold based on desired detectable event length

        :param int event_length: Maximum detectable event length
        '''
        indices = np.arange(0, self.dims)
        M = np.full(indices.shape, event_length)
        self.threshold = np.sqrt(np.sum(np.power(M-indices, 2)) / self.dims)


    def _set_tof(self, neighbor_indices):
        '''
        Helper function to set TOF parameters

        :param numpy.ndarray neighbor_indices: (n_samples, n_neighbors) array of n_neighbors nearest neighbors
                                               to each point in time-embedded phase space
        '''
        self.neighbor_indices_ = neighbor_indices
        indices = np.tile(np.arange(0, neighbor_indices.shape[0]).reshape(-1, 1), (1, neighbor_indices.shape[1]))
        self.tofs_ = np.power(
            np.sum(
                np.power(
                    np.absolute(
                        np.add(indices, -neighbor_indices)),
                    self.q
                ),
                axis=-1
            ) / self.dims,
            1 / self.q
        )
        self.tof_detections_ = np.where(self.tofs_ < self.threshold)[0]


    def fit(self, data, times=None):
        '''
        Populate internal parameters based on input time series

        :param numpy.ndarray data: 1D time series to be processed
                                   If multi-dimensional, will be reshaped to (-1,)
        :param numpy.ndarray times: (Optional) Corresponding time labels (must have same first dimension as data)
        :return: time series and corresponding time labels (if provided); truncated if wrap=False
        :rtype: numpy.ndarray
        '''
        if times is not None and data.shape[0] != times.shape[0]:
            raise ValueError(
                f'Expected times {times.shape} to have the same number of entries as data {data.shape}')
        
        data = data.flatten()
        self._time_delay_embed(data)
        self.kNN.fit(self.get_embedded_data())
        self._set_tof(self.kNN.kneighbors(return_distance=False))
        self._set_truncated_data(data, times, self.tofs_.size)


    def get_outlier_indices(self):
        return self.tof_detections_

    
    def get_outlier_factors(self):
        return self.tofs_

    
    def get_neighbor_indices(self):
        return self.neighbor_indices_
