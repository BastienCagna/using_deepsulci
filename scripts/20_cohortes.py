import matplotlib.pyplot as plt
import argparse
import os.path as op
import numpy as np
from utils import load_cohorts


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

    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")

    cohorts_plot(load_cohorts(env_f), 'L')


if __name__ == "__main__":
    main()
