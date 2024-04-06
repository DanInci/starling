from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment
import numpy as np


class AutomatedLabelMapping:

    # strategy can be absolute or percentage
    def __init__(self, ref_label, pred_cluster, strategy='percentage'):
        assert len(ref_label) == len(pred_cluster)
        assert strategy == 'absolute' or strategy == 'percentage'

        self.ref_label = ref_label

        self.cost_matrix, row_labels, col_labels = self._compute_cost_matrix(ref_label, pred_cluster)
        self.cluster_label_map = self._get_cluster_mapping(self.cost_matrix, row_labels, col_labels)

    def get_pred_labels(self, pred_cluster):
        return list(map(lambda c: self.cluster_label_map.get(c, str(c)), pred_cluster))

    def _compute_cluster_label_map(self, labels, clusters, strategy):
        zipped = list(zip(labels, clusters))
        grouped_by_cluster = defaultdict(list)

        for label, cluster in zipped:
            grouped_by_cluster[cluster].append(label)

        # for each cluster, the assigment is made to the label that is the most common as its reference
        cluster_label_map = {}
        if strategy == 'absolute':
            cluster_label_map = {
                cluster: Counter(labels_in_cluster).most_common(1)[0][0]
                for cluster, labels_in_cluster in grouped_by_cluster.items()
            }
        elif strategy == 'percentage':
            labels_total = Counter(labels)
            cluster_label_map = {
                cluster: get_label_with_highest_percent(Counter(labels_in_cluster), labels_total)
                for cluster, labels_in_cluster in grouped_by_cluster.items()
            }

        return cluster_label_map

    def _compute_cost_matrix(self, ref_labels, pred_clusters):
        # Create a matrix to store the count
        unique_ref_labels = np.unique(ref_labels)
        unique_pred_clusters = np.unique(pred_clusters)
        count_matrix = np.zeros((len(unique_ref_labels), len(unique_pred_clusters)), dtype=int)

        # Iterate over the indices and update the matrix
        for i in range(len(ref_labels)):
            count_matrix[unique_ref_labels == ref_labels[i], unique_pred_clusters == pred_clusters[i]] += 1

        cost_matrix = np.max(count_matrix, axis=1)[:, np.newaxis] - count_matrix

        return cost_matrix, unique_ref_labels, unique_pred_clusters

    def _get_cluster_mapping(self, cost_matrix, row_labels, col_labels):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        cluster_map = {}
        for i, j in zip(row_ind, col_ind):
            to_label = row_labels[i]
            from_label = col_labels[j]
            cluster_map[from_label] = to_label

        return cluster_map


def get_label_with_highest_percent(labels_in_cluster, labels_total):
    results = {}

    for label in labels_in_cluster.keys():
        results[label] = labels_in_cluster[label] / labels_total[label]

    return max(results, key=results.get)
