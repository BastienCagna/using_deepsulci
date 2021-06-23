import argparse
import os.path as op
from os import makedirs
from soma import aims


def main():
    parser = argparse.ArgumentParser(description='Remove all names in a graph except one')
    parser.add_argument('-c', dest='cohorts', type=str, nargs='+', default=None, required=False,
                        help='Cohort names')
    parser.add_argument('-s', dest='sulci_name', type=str, default=None, required=False,
                        help='The only sulci name to keep')

    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    cohorts_dir = op.join(env['working_path'], "cohorts")

    modif_dir = op.join(env["working_path"], "modified_graphs")
    makedirs(modif_dir, exist_ok=True)

    subjects = set()
    for c in args.cohorts:
        cohort = Cohort(from_json=op.join(cohorts_dir, "cohort-" + c + ".json"))
        for sub in cohort.subjects:
            subjects.add(sub)

        for sub in subjects:
            filter_sulci_names(sub.graph)


if __name__ == "__main__":
    # main()
    ig = "/neurospin/dico/data/bv_databases/human/archi/t1-1mm-1/013/t1mri/default_acquisition/default_analysis/folds/3.3/session1_manual/L013_session1_manual.arg"
    og = "/var/tmp/L013_session1_manual_only-S.intraCing._left.arg"
    to_keep = ["S.intraCing._left"]
    filter_sulci_names(ig, og, to_keep)
