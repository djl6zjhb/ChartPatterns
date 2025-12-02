# Class generated from ChatGPT to implement a Walk-Forward Split cross-validation strategy.
# I identified a package that would do this (mlfinlab) but it is not compatible with Python 3.12 yet.
# This was a necessary step to prevent data leakage in my prediction pipeline.
# To ensure correct implementation, I used generative AI (ChatGPT) to create this class.                                                                                            

import numpy as np

class WalkForwardSplit:
    """
    Simple expanding-window, time-aware cross-validation splitter.

    Parameters
    ----------
    n_splits : int
        Number of folds (train/test pairs) to generate.
    test_size : int, optional
        Number of samples in each test fold. If None, it's inferred.
    min_train_size : int, optional
        Minimum number of samples in the first training fold. If None,
        it's inferred.
    """

    def __init__(self, n_splits=5, test_size=None, min_train_size=None, times=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.times = times

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices for train/test for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or DataFrame
        times : array-like of shape (n_samples,)
            Timestamps or sortable values that define chronological order.

        Yields
        ------
        train_idx, test_idx : np.ndarray
            Arrays of integer indices into X.
        """
        if self.times is None:
            raise ValueError("times must be provided for WalkForwardSplit")

        times = np.asarray(self.times)
        n_samples = len(times)

        # Sort by time
        order = np.argsort(times)
        ordered_times = times[order]

        # Infer test_size / min_train_size if not provided
        if self.test_size is None or self.min_train_size is None:
            # crude but reasonable defaults: leave roughly
            # (n_splits + 1) equal segments
            fold_size = n_samples // (self.n_splits + 1)
            if fold_size == 0:
                raise ValueError("Not enough samples for the requested n_splits.")
            test_size = fold_size
            min_train_size = fold_size
        else:
            test_size = self.test_size
            min_train_size = self.min_train_size

        for k in range(self.n_splits):
            train_end = min_train_size + k * test_size
            test_start = train_end
            test_end = test_start + test_size

            if test_end > n_samples:
                # Not enough data left for another full test window
                break

            train_idx = order[:train_end]
            test_idx = order[test_start:test_end]

            # Sanity: all train times < all test times
            assert ordered_times[train_idx].max() <= ordered_times[test_idx].min()

            yield train_idx, test_idx
