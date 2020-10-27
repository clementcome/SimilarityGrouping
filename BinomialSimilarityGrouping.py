import numpy as np
import scipy.stats as stats
from scipy.special import binom
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, single


class BSG:
    """
    Performs similarity grouping on data that is spatially distributed
    and has also a Binomial distributed value associated to it.
    """

    def __init__(self, n_estimation, p, delta_epsilon, n_cluster_max):
        self.n_estimation_ = n_estimation
        self.p_ = p
        self.delta_epsilon_ = delta_epsilon
        self.n_cluster_max_ = n_cluster_max
        s = np.arange(n_estimation + 1)
        self.proba_array_ = stats.binom.pmf(s, n_estimation, p)
        self.linkage_matrix_ = None

    def dist_proba(self, x, y):
        x, y = int(min(x, y)), int(max(x, y))
        return np.sum(self.proba_array_[x:y])

    def fit_epsilon(self, X, y, epsilon):
        dist = squareform(pdist(X))
        dist_distribution = squareform(pdist(y.reshape(-1, 1), metric=self.dist_proba))
        link_matrix = epsilon * (1 - dist_distribution) >= dist
        return link_matrix

    def fit_hierarchy(self, X, y):
        dist = pdist(X)
        dist_distribution = pdist(y.reshape(-1, 1), metric=self.dist_proba)
        combined_dist = dist / (1 - dist_distribution)
        self.linkage_matrix_ = single(combined_dist)
        return self.linkage_matrix_

    def predict_hierarchy(self, max_cluster=10):
        self.labels_ = fcluster(
            self.linkage_matrix_, t=max_cluster, criterion="maxclust"
        )
        return self.labels_