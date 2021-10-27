from re import A
from using_deepsulci.anatomist import Anatomist
from using_deepsulci.utils.html import save_html_page
from os import listdir, makedirs
import os.path as op
from dico_toolbox.database import BVDatabase
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np


def main():
    # eval_dir = "/neurospin/dico/pauriau/data/training/evaluations/RMacaques16/RMacaques16_cv0"
    # output_dir = "/neurospin/dico/bcagna/data/pauriau/training/evaluations/RMacaques16/RMacaques16_cv0"

    eval_dir = "/neurospin/dico/pauriau/data/training/evaluations/RAllChimps60ESDLR/RAllChimps60ESDLR_cv0"
    output_dir = "/neurospin/dico/bcagna/data/pauriau/training/evaluations/RAllChimps60ESDLR/RAllChimps60ESDLR_cv0"

    hemi = op.split(eval_dir)[-1][0]

    if "Macaques" in eval_dir:
        space = "macaques"
        db_paths = ["/neurospin/dico/data/bv_databases/macaque/marsMacaques"]
    elif "Baboons" in eval_dir:
        space = "baboons"
        db_paths = ["/neurospin/dico/data/bv_databases/baboons/marsBaboons"]
    elif "Chimp" in eval_dir:
        space = "chimpenzes"
        db_paths = ["/neurospin/dico/data/bv_databases/chimps/3T_New",
                    "/neurospin/dico/data/bv_databases/chimps/Yerkes_15_RLCorrect"]
    else:
        raise ValueError("Can not find the space")

    print("Outdir:", output_dir)
    print("Hemi:", hemi)
    print("Space:", space)
    print("DB paths:", db_paths)

    dbs = list(BVDatabase(path) for path in db_paths)
    mesh_templ = "*/[sub]/*/*/*/segmentation/mesh/[sub]_[hemi]white.gii"

    graphs = []
    for f in listdir(eval_dir):
        if f.endswith(".arg"):
            graphs.append(op.join(eval_dir, f))
    print("Graphs count:", len(graphs))

    makedirs(output_dir, exist_ok=True)

    subnames = []
    subfiles = []
    whitemeshs = []
    ana = Anatomist()
    for graph in tqdm(graphs):
        mesh = None
        splits = op.split(graph)[1][1:].split('_')
        for s in range(len(splits)-1):
            sub = "_".join(splits[:s])
            mesh = None
            for db in dbs:
                meshs = db.get_from_template(
                    mesh_templ, sub=sub, hemi=hemi)
                if len(meshs) > 0:
                    mesh = meshs[0]
                    whitemeshs.append(mesh)
                    subnames.append(sub)
                    break
            if mesh is not None:
                break

        if mesh is None:
            print("No mesh for", graph)
        else:
            ff = []
            for ptw in ['left_to_right', 'right_to_left']:
                fname, _ = op.splitext(op.split(graph)[1])
                out_f = op.join(output_dir, fname + '_' + ptw + '.jpg')
                ana.labelled_graph_snapshot(mesh, graph, ptw, out_f)
                ff.append(op.split(out_f)[1])
            subfiles.append(ff)

    csv = pd.read_csv(graphs[0][:-4] + '_scores.csv')
    labels = set()
    for k in csv.keys():
        if k.startswith('acc_'):
            labels.add(k[4:])
    labels = list(labels)
    # scores = {k: [] for k in ['subject', 'label', 'Elocal']}

    plots = []
    esis = []
    for g, graph in enumerate(graphs):
        csv = pd.read_csv(graph[:-4] + '_scores.csv')
        scores = []
        for label in labels:
            scores.append(csv['Elocal_' + label][0])

        fig = plt.figure(figsize=(8, 4))
        plt.bar(range(len(labels)), scores)
        plt.xticks(range(len(labels)), labels, rotation=80)
        plt.title("global ESI: {:02f}%".format(float(csv['ESI']*100)))
        fname, _ = op.splitext(op.split(graph)[1])
        out_f = fname + '_scores.jpg'
        plots.append(out_f)
        fig.savefig(op.join(output_dir, out_f))
        esis.append(float(csv['ESI']))

    html = '<h1>Automatic labelling on test set</h1>'
    html += '<p><b>Source:</b>' + eval_dir + '</p>'
    html += '<p><b>Average global ESI:</b> {:.02f}% (+/- {:.02}%)</p>'.format(
        np.mean(esis), np.std(esis))
    html += '<table class="table"><tr><th>Subject</th><th colspan=2>Right side</th><th></th></tr>'
    for s, sub in enumerate(subnames):
        html += '<tr><th>{}</th><td><img src="./{}"/></td><td><img src="./{}"/></td><td><img src="./{}"/></td></tr>'.format(
            sub, subfiles[s][0], subfiles[s][1], plots[s]
        )
    html += '</table>'

    save_html_page(html, op.join(output_dir, "report.html"))


if __name__ == "__main__":
    main()
