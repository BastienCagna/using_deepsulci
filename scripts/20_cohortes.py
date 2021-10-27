import matplotlib.pyplot as plt
from using_deepsulci.cohort import Cohort
import argparse
import json
import os.path as op
from os import listdir
import numpy as np


def cohorts_plot(cohorts, hemi):
    n_rows = len(cohorts)
    subjects = []
    for c in cohorts:
        if c.name[0].isupper() and c.name.endswith(hemi):
            print("Add", c.name)
            subjects.extend(set(s.name for s in c.subjects))
    n_cols = len(subjects)

    img = np.zeros((n_rows, n_cols))
    for ic, c in enumerate(cohorts):
        for j, sub in enumerate(subjects):
            if sub in c:
                img[ic, j] = 1

    fig = plt.figure(figsize=(12, 6))
    plt.imshow(img, interpolation="nearest", aspect="auto")
    plt.xticks(range(len(subjects)), subjects, rotation=60)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Create cohorts files (.json)')
    parser.add_argument('-e', dest='env', type=str, default=None, help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    c_dir = op.join(env['working_path'], "cohorts")
    cohorts = list(Cohort(from_json=op.join(c_dir, f)) for f in listdir(c_dir))
    cohorts_plot(cohorts, 'L')


if __name__ == "__main__":
    main()
