import os.path as op
import json

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from soma import aims
from deepsulci.deeptools.dataset import extract_data
from deepsulci.sulci_labeling.method.unet import UnetSulciLabeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from using_deepsulci.cohort import Cohort


def main():
    wk_dir = "/neurospin/dico/bcagna/data/deepsulci_basic_learning"
    model_f = wk_dir + "/models/cohort-pclean50A_hemi-L_model-unet3d_d00b01/run-01/cohort-pclean50A_hemi-L_model-unet3d_d00b01_model.mdsm"
    cohort_name = 'PClean_hemi-L'
    out_f = '/var/tmp/tmp_data_sulci.csv'
    vs = 2

    cfile = wk_dir + '/cohorts/cohort-' + cohort_name + '.json'
    cohort = Cohort(from_json=cfile)

    graphs_f = [s.graph for s in cohort.subjects]
    model_params_f = model_f[:-11] + '_params.json'

    params = json.load(open(model_params_f, 'r'))
    ss_list = params["sulci_side_list"]

    n_graphs = len(graphs_f)
    n_labels = len(ss_list)
    print('{} labels'.format(n_labels))

    model = UnetSulciLabeling(
        ss_list, num_filter=64, batch_size=1, dropout=0, cuda=-1)
    model.load(model_f, dropout=0)

    results = {}
    for ss in ss_list:
        results[ss + '_p'] = []
        results[ss + '_e'] = []

    # for ig, gfile in enumerate(graphs_f):
    #     print("Labelling graph {}/{}".format(ig+1, n_graphs))
    #     data = extract_data(aims.read(gfile), voxel_size=vs, filter_key='name',
    #                         filter_list=ss_list)
    #     ytrue, ypred, prob = model.labeling(gfile, data['bck'], data['names'])
    #
    #     labels = np.argmax(prob, axis=1)
    #     labels_prob = np.exp(np.max(prob, axis=1))
    #
    #     for s, ss in enumerate(ss_list):
    #         avg_prob = np.nan_to_num(np.mean(labels_prob[labels == s]))
    #         error_rate = np.sum(np.array(ytrue) == np.array(ypred)) / len(ypred)
    #         results[ss + '_p'].append(avg_prob)
    #         results[ss + '_e'].append(error_rate)
    # df = pd.DataFrame(results)
    # df.to_csv(out_f, index=False)

    results = pd.read_csv(out_f)
    clf = LinearRegression()
    x = np.linspace(0, 1, 2)
    lin_preds = {}
    r = []
    coefs = {}
    for ss in ss_list:
        if len(results[ss + '_p']) > 0:
            xs = np.atleast_2d(results[ss + '_p']).T
            clf.fit(xs, results[ss + '_e'])
            lin_preds[ss] = clf.predict(np.atleast_2d(x).T)
            r.append(r2_score(results[ss + '_e'], clf.predict(xs)))
            coefs[ss] = [clf.coef_[0], clf.intercept_]
        else:
            r.append(0)
    r = np.array(r)

    sort_idx = np.argsort(r)
    r = r[sort_idx]
    ss_list = np.array(ss_list)[sort_idx]

    v = 12
    fig, axes = plt.subplots(6, v, figsize=(24, 12), sharex=True, sharey=True)
    for i in range(6):
        for j in range(v):
            idx = i * v + j
            if idx in range(n_labels):
                ss = ss_list[idx]

                ax = axes[i, j]
                ax.scatter(results[ss + '_p'], results[ss + '_e'], s=4)
                if ss in lin_preds.keys():
                    ax.plot(x, lin_preds[ss], '--k', linewidth=.8)
                    ax.text(.1, .1, "{:.03f}".format(r[idx]))
                    ax.text(.1, .2, "{:.3f}*x + {:.03f}".format(coefs[ss][0], coefs[ss][1]))
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_title(ss)
                ax.grid()

    plt.tight_layout()
    fig.savefig(op.join(wk_dir + '/figures/labelling_accuracy_vs_label_probability.svg'))
    plt.show()


if __name__ == "__main__":
    main()
