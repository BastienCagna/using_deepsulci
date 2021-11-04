from os import listdir
import os.path as op
from collections import Counter
from tqdm import tqdm
import pandas as pd
from soma import aims

from using_deepsulci.utils.dataframe_tableview import show_dataframe


def check_graph(graph_f, h, key='name'):
    if op.exists(graph_f):
        graph = aims.read(graph_f)
    else:
        print(graph_f, "is missing")
        return {}

    # Count each label
    labels = []
    for v in graph.vertices():
        if key in v.keys():
            labels.append(v[key])
    counts = Counter(labels)

    wrongs = {}
    for l in set(labels):
        if (h == 'L' and l[-6:] == '_right') or \
                (h == 'R' and l[-5:] == '_left'):
            wrongs[l] = counts[l]
    return wrongs


def main():
    db = "/neurospin/dico/data/bv_databases/macaque/marsMacaques/PRIME_DE"
    g = db + "/[sub]/" \
             "t1mri/default_acquisition/default_analysis/folds/3.1/" \
             "default_session_KK/[hemi][sub]_default_session_KK.arg"

    # List subjects
    subjects = []
    for f in listdir(db):
        if op.isdir(op.join(db, f)):
            subjects.append(f)

    # report = {k: [] for k in ['subject', 'hemi', 'label', 'count']}
    # for h in ['L', 'R']:
    #     for s in tqdm(subjects):
    #         wrongs = check_graph(g.replace('[sub]', s).replace('[hemi]', h), h)
    #         for l in wrongs.keys():
    #             report['subject'].append(s)
    #             report['hemi'].append(h)
    #             report['label'].append(l)
    #             report['count'].append(wrongs[l])
    #
    # df = pd.DataFrame(report)
    # df.to_csv('/var/tmp/check_labels.csv')
    df = pd.read_csv('/var/tmp/check_labels.csv')
    show_dataframe(df)


if __name__ == "__main__":
    main()
