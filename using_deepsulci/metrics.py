import numpy as np
from scipy.spatial.distance import euclidean


def average_proba(graphs, data, model, ss_list):
    """ Get a score by graph which is the average label probability """
    scores = []
    for graph, data in zip(graphs, data):
        _, _, y_scores = model.labeling(
            graph, data['bck'], ['unknown'] * len(data['bck']),
            n_iter=0)
        scores.append(np.mean(np.max(y_scores, axis=1)))
    return scores


def median_proba(graphs, data, model, ss_list):
    # Get a score by graph which is the average label probability
    scores = []
    for graph, data in zip(graphs, data):
        _, _, y_scores = model.labeling(
            graph, data['bck'], ['unknown'] * len(data['bck']),
            n_iter=0)
        scores.append(np.median(np.max(y_scores, axis=1)))
    return scores


def intersubjects_probability_matrix(graphs, data, model, ss_list):
    n_graphs = len(graphs)
    n_labels = len(ss_list)

    probas = []
    for graph, data in zip(graphs, data):
        _, _, prob = model.labeling(
            graph, data['bck'], ['unknown'] * len(data['bck']))

        labels = np.argmax(prob, axis=1)
        labels_prob = np.exp(np.max(prob, axis=1))

        p_vect = np.nan_to_num(list(np.mean(labels_prob[labels == s])
                                    for s in range(n_labels)))
        probas.append(p_vect)

    # TODO: Verify in RSA toolbox if there is a faster way to compute it
    dist_mat = np.zeros((n_graphs, n_graphs), dtype=float)
    # norms = np.zeros((n_graphs,), dtype=float)
    # zeros = np.zeros((n_labels,))
    for i, pi in enumerate(probas):
        # norms[i] = euclidean(pi, zeros)
        for j, pj in enumerate(probas[i + 1:]):
            d = euclidean(pi, pj)
            dist_mat[i, j + i + 1] = d
            dist_mat[j + i + 1, i] = d
    # avg_norm = np.mean(norms)
    # avg_dist = np.mean(dist_mat)

    # a = avg_norm / avg_dist
    # scores = norms - a * np.mean(dist_mat, axis=1)
    # # scores = list(norms[i] - a * np.mean(dist_mat[i]) for i in range(n_graphs))
    return dist_mat


def certainty_measure(graphs, data, model, ss_list):
    scores = []
    n_outs = len(ss_list)
    var_max = np.power(1 - 1 / n_outs, 2) / n_outs
    for graph, data in zip(graphs, data):
        _, _, prob = model.labeling(
            graph, data['bck'], ['unknown'] * len(data['bck']))
        # Normalized variance (range is then between 0 and 1)
        # TODO: take care about sulci size (or topology)
        scores.append(np.median(np.var(np.exp(prob), axis=1) / var_max))
    return scores


def similarity_measure(similarity_matrix, train_index, available_index):
    n_avalailable = len(available_index)
    n_train = len(train_index)
    n_all = n_avalailable + n_train
    r1, r2 = (n_avalailable - 1) / n_all, (n_train + 1) / n_all

    scores = []
    for sim_vect in similarity_matrix[available_index]:
        sum_train = np.sum(sim_vect[train_index])
        sum_avail = np.sum(sim_vect[available_index])
        scores.append(np.abs(r1 * sum_train - r2 * sum_avail))

    return scores


def random_scores(graphs, data, model, ss_list):
    return np.random.rand((len(graphs, )))
