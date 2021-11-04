import argparse
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from utils import extend_folds
import pandas as pd
import json
from utils import sulci_list_from_evaluation


def log_path(env, train_cohort, test_cohort, modelname, run_idx, hemi):
    model = "cohort-{}_hemi-{}_model-{}".format(train_cohort, hemi, modelname)
    run = 'run-{:02d}'.format(run_idx)
    training_log = op.join(env['working_path'], 'models', model, run,
                           model + '_log.csv')

    eval_log = op.join(env['working_path'], 'evaluations', model, run,
                       "cohort-" + test_cohort + '_hemi-' + hemi + '.csv')

    return training_log, eval_log


def load_evaluation_scores(log_f, score="ESI"):
    df = pd.read_csv(log_f)
    graphs = df['Unnamed: 0.1']

    # List labels
    labels = []
    for key in df.keys():
        if key.startswith(score + "_"):
            labels.append(key[len(score)+1:])

    scores = np.empty((len(graphs), len(labels)), dtype=float)
    for ig, graph in enumerate(graphs):
        for il, label in enumerate(labels):
            scores[ig, il] = df[score + '_' + label][ig]
    return scores, np.array(graphs), np.array(labels)


def scores_by_run_and_split(test_logs, title, save_as=None):

    avg_scores = []
    for log in test_logs:
        avg_scores.append(np.mean(load_evaluation_scores(log, score='ESI')[0]))

    names = []
    for f in test_logs:
        ff = op.split(f)[1].split('_')
        names.append(ff[0][7:] + '(' + ff[1][5] + ')')

    fig = plt.figure()
    plt.scatter(np.arange(len(avg_scores)), avg_scores)
    plt.ylim((0.011, 0.016))
    plt.xticks(
        np.arange(len(avg_scores)),
        names,
        rotation=45
    )
    plt.title(title)

    if save_as:
        fig.savefig(save_as)


def scores_by_train(datasets, title, save_as=None, metric='ESI'):
    raw_scores = {}
    for setkey in datasets.keys():
        raw = []
        for f in datasets[setkey]:
            if f:
                test_scores = load_evaluation_scores(f, score=metric)[0]
                raw.append(np.sum(test_scores, axis=1))
            else:
                raw.append([])
        raw_scores[setkey] = raw

    # plt.hist(raw_scores['PClean'], bins=100)
    # plt.show()
    # names = []
    # for f in test_logs:
    #     ff = op.split(f)[1].split('_')
    #     names.append(ff[0][7:] + '(' + ff[1][5] + ')')

    colors = ['gold', 'limegreen', 'royalblue']
    ecolors = ['goldenrod', 'green', 'navy']
    pcolors = ['orange', 'darkgreen', 'indigo']

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    for iset, setkey in enumerate(datasets.keys()):
        for i in range(3):
            x = iset * 5 + i + 1

            yy = raw_scores[setkey][i]
            # plt.scatter(xx, yy, c=pcolors[i], s=.5, linewidths=.5, alpha=.05)
            if len(yy) > 0:
                yy = yy[~np.isnan(yy)]
                xx = x + np.random.rand(len(yy)) * 0.4 - 0.2
                parts = plt.violinplot(yy, [x], widths=.8, showextrema=False,
                                       quantiles=[.1, .5, .9])
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i])
                    # pc.set_edgecolor('black')
                    pc.set_alpha(1)
                    pc.set_zorder(3)
                parts['cquantiles'].set_color('black')
                parts['cquantiles'].set_zorder(4)
            #
            # y = np.mean(yy)
            # plt.plot([x-.4, x+.4], [y, y], alpha=.6, color=colors[i])
    plt.ylim((0., 1))
    plt.xticks(
        np.arange(2, len(list(datasets.keys())*5), step=5),
        list(datasets.keys())
    )
    plt.xlim(0, len(list(datasets.keys())*5))
    major_ticks = np.arange(0, 1, .1)
    minor_ticks = np.arange(0, 1, .05)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(axis='y', which='minor', alpha=0.2)
    ax.grid(axis='y', which='major', alpha=0.5)
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('Training cohort')

    if save_as:
        fig.savefig(save_as)


def esi_by_sulcus(models_f, save_as=None):
    sslist = sulci_list_from_evaluation(models_f[0])
    sslist.remove('unknown')
    for l in sslist:
        if l.startswith('ventricle'):
            sslist.remove(l)
    sslist = np.array(sslist)

    maxs = []
    means = []
    stds = []
    sizes = []
    for model_f in models_f:
        scr_arr, _, labels = load_evaluation_scores(model_f, score='Elocal')
        s_arr = load_evaluation_scores(model_f, score='s')[0]
        max_scr = []
        mean_scr = []
        std_scr = []
        size_scr = []
        for l in sslist:
            sel = labels == l
            if np.sum(sel):
                max_scr.append(np.max(scr_arr[:, sel]))
                mean_scr.append(np.mean(scr_arr[:, sel]))
                std_scr.append(np.std(scr_arr[:, sel]))
                size_scr.append(np.mean(s_arr[:, sel]))
            else:
                max_scr.append(np.nan)
                size_scr.append(np.nan)
                mean_scr.append(np.nan)
                std_scr.append(np.nan)
        maxs.append(max_scr)
        means.append(mean_scr)
        stds.append(std_scr)
        sizes.append(size_scr)

    avg_sizes = np.mean(sizes, axis=0)
    sorted_idx = np.argsort(-avg_sizes)
    avg_sizes = avg_sizes[sorted_idx]
    sslist = sslist[sorted_idx]
    maxs = list(np.array(arr)[sorted_idx] for arr in maxs)
    means = list(np.array(arr)[sorted_idx] for arr in means)
    stds = list(np.array(arr)[sorted_idx] for arr in stds)

    values = 100*np.array(means)

    ticks = []
    for l in sslist:
        if l.endswith('._left'):
            ticks.append(l[:-6])
        elif l.endswith('_left'):
            ticks.append(l[:-5])
        else:
            ticks.append(l)
    colors = ['red', 'blue']
    y = np.arange(0, len(avg_sizes))

    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},
                           figsize=[10, 14], sharey=True)
    fig.subplots_adjust(wspace=.05, hspace=0)

    for i, arr in enumerate(values):
        ax[0].barh(y, arr, align='center', color=colors[i], alpha=.5,
                   zorder=3)
        # for j, l in enumerate(sslist):
        #     v = j - .2 + i * .4
        #     ax[0].plot([arr[j] - stds[i][j], arr[j] + stds[i][j]], [v, v],
        #                c=colors[i], zorder=4, alpha=.5)

    # ax[0].set_xlim(0, 100)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(ticks)
    ax[0].set_xlabel('$E_{Local}^{Mean}$ (%)')
    # ax[0].set_xlabel('$E_{Local}^{Max}$ (%)')
    ax[0].set_facecolor('whitesmoke')
    ax[0].grid(color='white')
    ax[0].text(20, 1, "Overall mean: {:.01f}% (+/- {:.01f}%)".format(
        np.mean(values[np.isfinite(values)]), np.std(values[np.isfinite(values)])
    ))
    ax[0].legend(list(op.split(f)[0].split(op.sep)[-2].split('_')[0][7:]
                      for f in models_f))

    ax[1].barh(y, avg_sizes, .9, align='center', color='k', zorder=3)
    ax[1].tick_params(left=False, labelleft=False)
    ax[1].set_ylim((-.5, len(sslist)-.5))
    ax[1].set_facecolor('whitesmoke')
    ax[1].grid(color='white')
    ax[1].set_xlabel('Average sulcus size ($mm^{3}$)')

    if save_as:
        fig.savefig(save_as)


def main():
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    args = parser.parse_args()

    env_file = args.env if args.env else \
        op.join(op.dirname(op.realpath(__file__)), 'env.json')
    env = json.load(open(env_file, 'r'))

    folds = [
            # ('pclean12*', ['p25a25*']),
            # ('archi12*', ['p25a25*']),
            ('HCP', ['Archi', 'PClean']),
            # ('pclean50*', ['pclean12*', 'archi12*']),
            # ('archi50*', ['pclean12*', 'archi12*']),
            # ('p25a25*', ['pclean12*', 'archi12*']),
            ('PClean', ['Archi']),
            ('Archi',  ['PClean']),
            ('p54a70*', ['pclean08*', 'archi08*']),
        ]


    r = 1
    h = 'L'
    m = 'unet3d_d00b01'
    s = 'A'

    # datasets = {
    #     "PClean12": [
    #         log_path(env, "pclean12" + s, "pclean50" + s, m, r, h)[1],
    #         log_path(env, "pclean12" + s, "archi50" + s, m, r, h)[1],
    #         log_path(env, "pclean12" + s, "hcp50" + s, m, r, h)[1],
    #     ],
    #     # "Archi12": [
    #     #     log_path(env, "archi12" + s, "pclean50" + s, m, r, h)[1],
    #     #     log_path(env, "archi12" + s, "archi50" + s, m, r, h)[1],
    #     #     log_path(env, "archi12" + s, "hcp50" + s, m, r, h)[1],
    #     # ],
    #     # "HCP12": [
    #     #     log_path(env, "hcp12" + s, "pclean50" + s, m, r, h)[1],
    #     #     log_path(env, "hcp12" + s, "archi50" + s, m, r, h)[1],
    #     #     log_path(env, "hcp12" + s, "hcp50" + s, m, r, h)[1],
    #     # ],
    #     "PClean": [
    #         None,
    #         log_path(env, "PClean", "Archi", m, r, h)[1],
    #         log_path(env, "PClean", "HCP", m, r, h)[1]
    #     ],
    #     "Archi": [
    #         log_path(env, "Archi", "PClean", m, r, h)[1],
    #         None,
    #         log_path(env, "Archi", "HCP", m, r, h)[1]
    #     ],
    #     "HCP": [
    #         log_path(env, "HCP", "PClean", m, r, h)[1],
    #         log_path(env, "HCP", "Archi", m, r, h)[1],
    #         None
    #     ],
    #     "PClean50" + s: [
    #         log_path(env, "pclean50" + s, "pclean12" + s, m, r, h)[1],
    #         log_path(env, "pclean50" + s, "archi12" + s, m, r, h)[1],
    #         log_path(env, "pclean50" + s, "hcp12" + s, m, r, h)[1],
    #     ],
    #     "Archi50" + s: [
    #         log_path(env, "archi50" + s, "pclean12" + s, m, r, h)[1],
    #         log_path(env, "archi50" + s, "archi12" + s, m, r, h)[1],
    #         log_path(env, "archi50" + s, "hcp12" + s, m, r, h)[1],
    #     ],
    #     # "HCP50" + s: [
    #     #     log_path(env, "hcp50" + s, "pclean12" + s, m, r, h)[1],
    #     #     log_path(env, "hcp50" + s, "archi12" + s, m, r, h)[1],
    #     #     log_path(env, "hcp50" + s, "hcp12" + s, m, r, h)[1],
    #     # ],
    #     "p25a25" + s: [
    #         log_path(env, "p25a25" + s, "pclean12" + s, m, r, h)[1],
    #         log_path(env, "p25a25" + s, "archi12" + s, m, r, h)[1],
    #         log_path(env, "p25a25" + s, "hcp12" + s, m, r, h)[1],
    #     ],
    #     "p54a70" + s: [
    #         log_path(env, "p54a70" + s, "pclean08" + s, m, r, h)[1],
    #         log_path(env, "p54a70" + s, "archi08" + s, m, r, h)[1],
    #         log_path(env, "p54a70" + s, "hcp08" + s, m, r, h)[1],
    #     ],
    #     "p54a70h68 (192)" + s: [
    #         log_path(env, "p54a70h68" + s, "pclean08" + s, m, r, h)[1],
    #         log_path(env, "p54a70h68" + s, "archi08" + s, m, r, h)[1],
    #         log_path(env, "p54a70h68" + s, "hcp08" + s, m, r, h)[1],
    #     ],
    # }
    #
    # scores_by_train(datasets, "Test accuracy over training cohorts",
    #     save_as=op.join(env["working_path"], "figures", "esi_vs_training_cohort.svg"),
    #     metric='ESI')
    esi_by_sulcus([
        log_path(env, "pclean50" + s, "pclean12" + s, m, r, h)[1],
        log_path(env, "archi50" + s, "pclean12" + s, m, r, h)[1]
    ], save_as=op.join(env["working_path"], "figures", "esi_local_vs_label.svg"))
    plt.show()


if __name__ == "__main__":
    main()
