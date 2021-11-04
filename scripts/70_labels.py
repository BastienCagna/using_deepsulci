import argparse
import matplotlib.pyplot as plt
from utils import load_cohorts
import os.path as op
from soma import aims


def labels_of_a_cohort(cohort):
    plt.figure()
    plt.imshow()
    pass


def labels_of_cohorts(cohorts):
    n_row = len(cohorts)
    n_cols =
    plt.figure()
    plt.imshow()
    pass


def main():
    parser = argparse.ArgumentParser(description='Create cohorts files (.json)')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    args = parser.parse_args()

    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")

    labels_of_cohorts(load_cohorts(env_f), 'L')


if __name__ == "__main__":
    main()
