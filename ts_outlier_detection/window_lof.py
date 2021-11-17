import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ts_outlier_detection.time_series_outlier import TimeSeriesOutlier

class WindowedLocalOutlierFactor(TimeSeriesOutlier):
    def __init__(self, n_neighbors=20, crit_lof=1.0, crit_sigma=None, **kwargs):
        '''
        Detects temporal outliers in one-dimensional time series data
        using a sliding window spatial embedding scheme and Local Outlier Factor

        :param int n_neighbors:  (Optional) Number of nearest neighbors to consider in outlier factor calculation (default 20)
        :param float crit_lof:   (Optional) Any point with an LOF above this will be considered outlying (default 1.0)
        :param float crit_sigma: (Optional) Alternative to specifying crit_lof; number of sigmas from mean to consider a point outlying (overrides crit_lof)

        Remaining parameters are passed to sklearn.neighbors.LocalOutlierFactor
        '''
        super().__init__(**kwargs)
        self.clf = LocalOutlierFactor(n_neighbors=n_neighbors, **self.unused_kwargs)
        self.crit_lof = crit_lof
        self.crit_sigma = crit_sigma


    def fit(self, data, times=None):
        '''
        Populate internal parameters based on input time series

        :param numpy.ndarray data: 1D time series to be processed
                                   (will be reshaped to (-1,) if multi-dimensional)
        :param numpy.ndarray times: (Optional) Corresponding time labels (must have same first dimension as data)
        :return: time series and corresponding time labels (if provided); truncated if wrap=False
        :rtype: numpy.ndarray
        '''
        if times is not None and data.shape[0] != times.shape[0]:
            raise ValueError(
                f'Expected times {times.shape} to have the same number of entries as data {data.shape}')
        data = data.flatten()
        self._time_delay_embed(data)
        self.clf.fit(self.get_embedded_data())
        self.lofs_ = -self.clf.negative_outlier_factor_
        self._set_truncated_data(data, times, self.lofs_.size)


    def get_outlier_indices(self):
        if self.crit_sigma is not None:
            mean_lof = np.mean(self.lofs_)
            lof_std = np.std(self.lofs_)
            self.crit_lof  = self.crit_sigma * lof_std + mean_lof
        return np.where(self.lofs_ > self.crit_lof)[0]

    
    def get_outlier_factors(self):
        return self.lofs_
    

    def get_neighbor_indices(self):
        return self.clf.kneighbors(return_distance=False)
