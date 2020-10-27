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

    def __init__(self, n_estimation):
        self.n_estimation_ = n_estimation
        self.linkage_matrix_ = None

    def dist_proba(self, xp, yp):
        x_min, x_max = int(min(xp[0], yp[0])), int(max(xp[0], yp[0]))
        return (
            np.sum(self.proba_array_dict_[xp[1]][x_min:x_max])
            + np.sum(self.proba_array_dict_[yp[1]][x_min:x_max])
        ) / 2

    def fit_hierarchy(self, X, y, p):
        n_sample, n_feature = X.shape
        y = y.reshape(-1, 1)
        if type(p) == float:
            p = np.repeat(p, n_sample)
        self.p_ = p.reshape(-1, 1)

        unique_p_arr = np.unique(self.p_)

        s = np.arange(self.n_estimation_ + 1)
        self.proba_array_dict_ = {
            unique_p: stats.binom.pmf(s, self.n_estimation_, unique_p)
            for unique_p in unique_p_arr
        }

        dist = pdist(X)
        dist_p = pdist(self.p_)

        yp = np.concatenate((y, self.p_), axis=1)
        dist_distribution = pdist(yp, metric=self.dist_proba)
        combined_dist = dist / (1 - dist_distribution) * (1 + dist_p)
        self.linkage_matrix_ = single(combined_dist)
        return self.linkage_matrix_

    def predict_hierarchy(self, max_cluster=10):
        self.labels_ = fcluster(
            self.linkage_matrix_, t=max_cluster, criterion="maxclust"
        )
        return self.labels_


# np.random.seed(42)
# n_estimation = 100
# n_cluster = 100
# p_list = [0.3, 0.5, 0.8]
# p_arr = np.random.choice(p_list, size=n_cluster)
# cluster_centers = np.random.rand(n_cluster, 2)
# binom_sample = np.array([stats.binom.rvs(n_estimation, p) for p in p_arr])
# bsg = BSG(n_estimation)
# linkage_matrix = bsg.fit_hierarchy(cluster_centers, binom_sample, p_arr)
# labels = bsg.predict_hierarchy(15)