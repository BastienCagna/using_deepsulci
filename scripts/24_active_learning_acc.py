import matplotlib.pyplot as plt
import argparse
import os.path as op
import json
from os import listdir
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Show accuracy curves')
    parser.add_argument('-m', dest='modelname', type=str,  required=True,
                        help='Model name')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    args = parser.parse_args()

    # Load environment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    outdir = op.join(env['working_path'], "models")
    args.modelname = args.modelname[:-2] if args.modelname[-2:] == "pa" else \
        args.modelname[:-1]
    colors = {'r': 'gray', 'pa': 'blue'}
    plt.figure()
    for sfx, method in [('r', 'random'), ('pa', 'proba_max')]:
        modelname = args.modelname + sfx
        model_dir = op.join(outdir, modelname)
        fname = modelname + "_acc_log.csv"
        lengths = []
        series = []
        for r in listdir(model_dir):
            if op.isdir(op.join(model_dir, r)):
                data = pd.read_csv(op.join(model_dir, r, fname))
                lengths.append(len(data['test_avg_accuracy']))
                series.append(data['test_avg_accuracy'])

        avg = []
        for i in range(max(lengths)):
            vect = []
            for a, acc in enumerate(series):
                if i <= lengths[a]:
                    vect.append(acc[i])
            avg.append(np.mean(vect))

        for acc in series:
            plt.plot(np.arange(len(acc)), acc, '*-', color=colors[sfx])
        plt.plot(np.arange(len(avg)), avg, linewidth=2, color=colors[sfx])
    plt.show()


if __name__ == "__main__":
    main()
