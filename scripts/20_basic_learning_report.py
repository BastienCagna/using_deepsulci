import argparse
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from utils import extend_folds
import pandas as pd
import json


def log_path(env, train_cohort, test_cohort, modelname, run_idx, hemi):
    model = "cohort-{}_hemi-{}_model-{}".format(train_cohort, hemi, modelname)
    run = 'run-{:02d}'.format(run_idx)
    training_log = op.join(env['working_path'], 'models', model, run,
                           model + '_log.csv')

    eval_log = op.join(env['working_path'], 'evaluation', model, run,
                       test_cohort + '.csv')

    return training_log, eval_log


def load_evaluation_scores(log_f, score="ESI"):
    df = pd.DataFrame(log_f)
    graphs = df['Unnamed: 0']

    # List labels
    labels = []
    for key in df.keys():
        if key.startwidth(score + "_"):
            labels.append(key[4:])

    scores = np.empty(len(graphs), len(labels))
    for ig, graph in enumerate(graphs):
        for il, label in enumerate(labels):
            scores[ig, il] = df[score + '_' + label][ig]
    return scores, graphs, labels


def scores_by_run_and_split(test_logs, title, save_as=None):

    avg_scores = []
    for log in test_logs:
        avg_scores.append(np.mean(load_evaluation_scores(log, score='ESI')))

    fig = plt.figure()
    plt.scatter(np.arange(len(avg_scores)), avg_scores)
    plt.xticks(
        np.arange(len(avg_scores)),
        list(op.split(f)[1] for f in test_logs)
    )
    plt.title(title)

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
            ('pclean12*', ['p25a25*']),
            ('archi12*', ['p25a25*']),
            ('pclean50*', ['pclean12*', 'archi12*']),
            ('archi50*', ['pclean12*', 'archi12*']),
            ('p25a25*', ['pclean12*', 'archi12*']),
            ('PClean', ['Archi']),
            ('Archi',  ['PClean']),
            ('p54a70*', ['pclean08*', 'archi08*']),
        ]

    for (train, tests) in folds:
        for h in ['L', 'R']:
            all_splits = extend_folds([(train, tests)])
            logs = []
            for r in [1, 2, 3]:
                for (tr, te) in all_splits:
                    logs.append(log_path(env, tr, te, 'unet3d_d00b01', r, h))
            scores_by_run_and_split(logs, train)
    plt.show()


if __name__ == "__main__":
    main()
