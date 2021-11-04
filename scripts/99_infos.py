"""
    A command line to print informations about cohortes, processings, results...
"""

import json
import argparse
import os.path as op
from os import listdir


def list_cohorts(env):
    cohorts_dir = op.join(env['working_path'], "cohorts")
    cnames = []
    for fname in listdir(cohorts_dir):
        if fname[:7] == "cohort-" and fname[-5:] == ".json":
            cnames.append(fname[7:-5])

    for i, cname in enumerate(sorted(cnames)):
        print(cname, end=' '*(40-len(cname)) if (i+1)%3 > 0 else '\n')
    print()


def main():
    parser = argparse.ArgumentParser(description='Infos')
    parser.add_argument('-e', dest='env', type=str, default=None, help="Configuration file")
    parser.add_argument('-c', dest='cohorts', action='store_const', const=True, default=False,
                        help="List cohorts")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    if args.cohorts:
        list_cohorts(env)


if __name__ == "__main__":
    main()
