import sys
import os.path as op
from os import listdir, makedirs
from soma import aims


def convert(graph_f, out_f, default="unknown"):
    graph = aims.read(graph_f)
    for v in graph.vertices():
        if 'label' in v.keys():
            if 'name' in v.keys() and v['name'] != default:
                raise ValueError("Already set name")
            v['name'] = v['label']
            v['label'] = default

    makedirs(op.split(out_f)[0], exist_ok=True)
    aims.write(graph, out_f)


def main():
    # files = sys.argv[1:]
    db = "/neurospin/dico/data/bv_databases/human/hcp_labeled/t1-0.7mm-1/"
    subjects = []
    for f in listdir(db):
        if op.isdir(op.join(db, f)):
            subjects.append(f)
    in_pat = db + "[sub]/t1mri/default_acquisition/default_analysis/folds/" \
                  "3.1/default_session_auto/[h][sub]_default_session_auto.arg"
    out_pat = db + "[sub]/t1mri/default_acquisition/default_analysis/folds/" \
                   "3.1/manually_corrected/[h][sub]_manually_corrected.arg"

    for s in subjects:
        for h in ['L', 'R']:
            print(s, h)
            in_f = in_pat.replace('[sub]', s).replace('[h]', h)
            out_f = out_pat.replace('[sub]', s).replace('[h]', h)
            convert(in_f, out_f)


if __name__ == "__main__":
    main()
