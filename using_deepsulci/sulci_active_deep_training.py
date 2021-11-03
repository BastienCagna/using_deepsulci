from __future__ import print_function
from __future__ import absolute_import
from random import weibullvariate
import traits.api as traits
import os
import os.path as op
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import sigraph
from deepsulci.sulci_labeling.method.unet import UnetSulciLabeling
from deepsulci.sulci_labeling.capsul.training import SulciDeepTraining, load_graphs, get_sulci_side_list, standarized_names, index_of
from deepsulci.sulci_labeling.analyse.stats import esi_score
import matplotlib.pyplot as plt
import time


def local_esi_score(y_true, y_pred, labels):
    '''
    Local ESI score
    '''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp, fp, fn, s = {}, {}, {}, {}
    esi_vect = []
    for ss in labels:
        names_ss = y_pred[y_true == ss]
        labels_ss = y_true[y_pred == ss]

        tp[ss] = float(len(names_ss[names_ss == ss]))
        fp[ss] = float(len(labels_ss[labels_ss != ss]))
        fn[ss] = float(len(names_ss[names_ss != ss]))

        if fp[ss] + fn[ss] + 2*tp[ss] != 0:
            esi = (fp[ss]+fn[ss]) / (fp[ss]+fn[ss]+2*tp[ss])
        else:
            esi = 0
        esi_vect.append(esi)

    return esi_vect


def scores_histogram(scores, title, save_as):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    plt.hist(scores, bins=np.linspace(0, 1, 40))
    plt.xlim(0, 1)
    plt.title(title)
    fig.savefig(save_as)
    print("Scores histogram saved in:", save_as)


def plot_dist_mat(mat, out_f, title):
    fig = plt.figure(figsize=(8, 8))
    plt.show(mat, interpolation="nearest", aspec="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    fig.savefig(out_f)


def certainty_measure(graphs, data, model, ss_list, weights=None):
    if weights is None:
        weights = np.ones((len(ss_list),))

    # all weight are 0
    weights = np.array(weights)
    print(weights)
    if len(weights) != len(ss_list):
        raise ValueError("Cannot apply weights of shape {} to sulci list of length {}.".format(
            weights.shape, len(ss_list)))

    scores = []
    n_outs = len(ss_list)
    var_max = np.power(1 - 1 / n_outs, 2) / n_outs
    for graph, data in zip(graphs, data):
        _, _, prob = model.labeling(
            graph, data['bck'], ['unknown'] * len(data['bck']))
        # Normalized variance (range is then between 0 and 1)
        # TODO: take care about sulci size (or topology)
        norm_var = np.var(np.exp(prob), axis=1) / var_max
        gweights = np.tile(weights, (norm_var.shape[0], 1))
        print("shapes:", norm_var.shape, gweights.shape)
        scores.append(np.average(norm_var, weights=gweights))
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


def probability_representativeness(avail_graphs, test_graphs, avail_data, test_probas, model,
                                   plot_mat_in=None):
    n_avail_graphs = len(avail_graphs)
    n_test_graphs = len(test_graphs)
    n_graphs = n_avail_graphs + n_test_graphs

    test_probas = np.array(test_probas)
    n_voxels, n_labels = test_probas.shape

    # Compute average probas obtained for each graph and label
    probas = np.empty((n_graphs, n_labels * n_voxels))
    for g, (graph, probas) in enumerate(zip(test_graphs, test_probas)):
        probas[g] = probas.flatten()

    for g, (graph, data) in enumerate(zip(avail_graphs, avail_data)):
        _, _, probas = model.labeling(
            graph, data['bck'], ['unknown'] * len(data['bck']))
        probas[n_test_graphs + g] = np.array(probas).flatten()
        # # labels = np.argmax(prob, axis=1)
        # labels_prob = np.exp(np.max(prob, axis=1))

        # for s in range(n_labels):
        #     # Average probability accross all the voxels labelized with s
        #     probas[g + n_test_graphs, s] = np.mean(labels_prob[labels == s])

    dist_mat = np.zeros((n_graphs, n_graphs), dtype=float)
    for i, pi in enumerate(probas):
        for j, pj in enumerate(probas[i + 1:]):
            d = euclidean(pi, pj)
            dist_mat[i, j + i + 1] = d
            dist_mat[j + i + 1, i] = d

    if plot_mat_in is not None:
        plot_dist_mat(dist_mat, plot_mat_in, "Probability distance")
    return similarity_measure(dist_mat, np.arange(0, n_test_graphs), np.arange(n_test_graphs, n_avail_graphs))


class SulciActiveDeepTraining(SulciDeepTraining):
    def __init__(self):
        super().__init__()
        self.add_trait('available_graphs', traits.List(
            traits.File(output=False), desc='Available graphs'))
        self.add_trait('test_graphs', traits.List(
            traits.File(output=False), desc='Graphs used to test the model'))
        self.add_trait('similarity_matrix', traits.List(
            traits.File(output=False), desc='Graphs similarity matrix'))
        self.add_trait('inclusion_amount', traits.Int(default=1,
                                                      desc='Number of graph to include at each active learning iteration.'))
        self.add_trait('max_iter', traits.Int(default=-1,
                                              desc='Maximum number of active learning passes.'))
        self.add_trait('inclusion_strategy', traits.String(default="proba_avg",
                                                           desc='Strategy to use to select graphs (proba_avg/random)'))
        self.add_trait('retrain_from_scratch', traits.Bool(default=False,
                                                           desc='Set to True if you want to entirely retrain the model at each '
                                                           'pass'))
        self.add_trait('acc_log_file', traits.File(
            output=True,
            desc='file (.csv) storing active learning info'))
        self.available_data = None
        self.test_data = None
        self.trained_method = None
        self.sulci_side_list = None
        self.test_probas = None
        self.test_accuracy = None
        self.iteration = None

    def loading_step(self, flt=None):
        # Load only if data have never been read.
        if not self.available_data and not self.sulci_side_list:
            print('Reading all available graphs')
            data = load_graphs(self.graphs + self.available_graphs,
                               flt, self.voxel_size)
            ss_list = get_sulci_side_list(data)
            print("Sulci side list is set to:")
            print(ss_list)
            print("({} labels)".format(len(ss_list)))
            self.sulci_side_list = ss_list
            self.available_data = data[len(self.graphs):]
        if not self.test_data:
            self.test_data = load_graphs(self.test_graphs, flt,
                                         self.voxel_size,
                                         self.sulci_side_list)

        dict_bck, dict_names, _ = super().loading_step(flt, self.sulci_side_list)
        return dict_bck, dict_names

    def inclusion_step(self):
        if len(self.available_graphs) == 0:
            return None

        # Get one score for each available graphs
        if self.inclusion_strategy == "probability_representativeness":
            fig_f = op.join(
                self.model_file[:-5] + "_distance_matrix_it{:03d}.svg".format(self.iteration))
            scores = probability_representativeness(
                self.available_graphs, self.graphs, self.available_data, self.test_probas, self.model,
                plot_mat_in=fig_f)
        elif self.inclusion_strategy == "certainty":
            scores = certainty_measure(
                self.available_graphs, self.available_data, self.model, self.model.sslist)
        elif self.inclusion_strategy == "weighted_certainty":
            # Mean error rate accross test subjects
            print(self.test_accuracy)
            print(self.test_accuracy.shape)
            weights = 1 - np.mean(self.test_accuracy, axis=0)
            scores = certainty_measure(
                self.available_graphs, self.available_data, self.model, self.model.sslist, weights)
        elif self.inclusion_strategy == "random":
            scores = np.random.rand((len(self.available_graphs, )))
        else:
            raise ValueError("Unrecognized inclusion strategy '" +
                             self.inclusion_strategy + "'")

        # Order by ascending score
        order = np.argsort(scores)
        n = min(self.inclusion_amount, len(scores))
        print('Selecting {} graphs using {} strategy'.format(
            n, self.inclusion_strategy))
        incl_graphs = np.array(self.available_graphs)[order][:n]

        scores = np.array(scores)[order]
        for ig, g in enumerate(incl_graphs):
            print(op.split(g)[1], scores[ig])

        # # TODO: verify that graphs are labelled
        # labelled = True
        # if not labelled:
        #     print("Please label those graphs and click on enter to continue.")
        # else:
        #     print("Including those graphs in the next iteration:")
        # for g in incl_graphs:
        #     print("\t-", op.split(g)[1])
        # if not labelled:
        #     input("")

        return incl_graphs, scores

    def evaluation_step(self):
        print("Evaluating trained model on the testing set...")

        dict_names = standarized_names(self.test_graphs, self.test_data,
                                       self.sulci_side_list)

        # voxel labeling
        scores = []
        probas = []
        for ig, graph in enumerate(self.test_graphs):
            data = self.test_data[ig]
            # data = {k: np.asarray(v) for k, v in data.items()}

            y_true, y_pred, prob = self.model.labeling(
                graph, data['bck'], dict_names[graph], n_iter=0)
            scores.append(
                1 - np.array(local_esi_score(y_true, y_pred, self.sulci_side_list)))
            probas.append(prob)
        self.test_probas = probas
        self.test_accuracy = np.array(scores)

        avg, std = np.mean(scores), np.std(scores)
        print("\tAverage ESI: {} (+/- {})".format(avg, std))
        return avg, std

    def _run_process(self):
        print("Start active learning process")
        if os.path.exists(self.translation_file):
            flt = sigraph.FoldLabelsTranslator()
            flt.readLabels(self.translation_file)
            trfile = self.translation_file
        else:
            trfile, flt = None, None
            print('Translation file not found.')

        # Load data
        dict_bck, dict_names = self.loading_step(flt)

        # init method
        self.sslist = [ss for ss in self.sulci_side_list if
                       not ss.startswith('unknown')
                       and not ss.startswith('ventricle')]
        self.model = UnetSulciLabeling(
            self.sulci_side_list, num_filter=64, batch_size=self.batch_size,
            dropout=self.dropout, cuda=self.cuda, translation_file=trfile,
            dict_bck=dict_bck, dict_names=dict_names)
        # fix learning rate / momentum
        self.model.lr = .025
        self.model.momentum = .9

        log = {k: [] for k in ["best_accuracy", "best_epoch",
                               "test_avg_accuracy", "test_std_accuracy",
                               "n_epochs", "n_samples", "duration"]}
        if self.max_iter > 0:
            max_iter = self.max_iter
        else:
            max_iter = int(
                np.floor(len(self.available_graphs) // self.inclusion_amount))
        for it in range(max_iter):
            self.iteration = it
            start = time.time()
            print("\n\n*** Iteration {}/{} ***".format(it + 1, max_iter))
            print("{} used graphs - {} available graphs".format(
                len(self.graphs), len(self.available_graphs)))

            # Train deep model
            # TODO: if retrain from scratch, new sulci names can be added
            # TODO: update the sulci side list if needed
            if self.retrain_from_scratch and op.exists(self.model_file):
                os.remove(self.model_file)
            self.training_step()

            # Eval on the test set
            avg_acc, std_acc = self.evaluation_step()

            log['best_accuracy'].append(self.model.best_acc)
            log['best_epoch'].append(self.model.best_epoch)
            log['test_avg_accuracy'].append(avg_acc)
            log['test_std_accuracy'].append(std_acc)
            log['n_epochs'].append(self.model.n_epochs)
            log['n_samples'].append(len(self.graphs))
            log['duration'].append(time.time() - start)
            df = pd.DataFrame(log)
            df.to_csv(self.acc_log_file, index=False)
            print("Log saved in:", self.acc_log_file)

            # Get new manually labelled graph(s)
            selection, scores = self.inclusion_step()

            title = self.inclusion_strategy + \
                " iteration {}/{}".format(it+1, max_iter)
            fig_file = self.acc_log_file[:-4] + \
                "_scores_it-{:04d}.svg".format(it+1)
            scores_histogram(scores, title, fig_file)

            if selection is None:
                break
            for graph in selection:
                self.graphs.append(graph)
                idx = index_of(graph, self.available_graphs)
                del self.available_graphs[idx]
                del self.available_data[idx]

            # TODO: avoid rereading all files from the disk but keep the loop
            #  that check that sulci side list are consistent
            dict_bck, dict_names = self.loading_step(flt)
            self.model.dict_bck = dict_bck
            self.model.dict_names = dict_names

        print("Exit from the active learning loop. Bye.")
