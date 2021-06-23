import json
import os.path as op
from os import makedirs
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import html_intro, html_outro

import plotly.express as px



def parse_sulci_list(keys):
    ss_list = set()
    for k in keys:
        sp = k.split('_')
        if len(sp) > 1:
            ss_list.add('_'.join(sp[1:]))
    return list(sorted(ss_list))


def subject_name_from_graph_name(name):
    return op.split(name)[1].split('_')[1:]


if __name__ == "__main__":
    eval_dir = "/neurospin/dico/bcagna/data/deepsulci_models_comparison/evaluations/"
    models = [
        'cohort-pclean50A_hemi-L_model-unet3d',
        'cohort-pclean50A_hemi-L_model-unet3d_ext-archi50A_hemi-L'
    ]
    cohort = 'cohort-pclean12A_hemi-L'
    out_dir = "/var/tmp/model_comp_report"

    html_f = op.join(out_dir, "models_comparison.html")
    fig_dir = op.join(out_dir, "figures")
    makedirs(fig_dir, exist_ok=True)

    print("Loading scores...")
    ss_list = None
    scores = {k: [] for k in ['model', 'subject', 'name', 'sens', 'spec']}
    for model in models:
        print(model)
        data = pd.read_csv(op.join(eval_dir, model, cohort + ".csv"))

        if not ss_list:
            ss_list = parse_sulci_list(data.keys())

        for ig, g in tqdm(enumerate(data['Unnamed: 0.1'])):
            s = subject_name_from_graph_name(g)
            for ss in ss_list:
                scores['model'].append(model)
                scores['subject'].append(s)
                scores['name'].append(ss)
                scores['sens'].append(data["sens_" + ss][ig])
                scores['spec'].append(data["spec_" + ss][ig])
    scores = {k: np.array(scores[k]) for k in scores.keys()}
    print("Done")

    print("Generating figures...")
    for ss in tqdm(ss_list):
        colors = ['g', 'm']
        fig = plt.figure(figsize=(4, 4))
        for m in models:
            specs = scores['spec'][scores['model'] == m]
            sens = scores['sens'][scores['model'] == m]
            plt.scatter(np.log10(specs), np.log10(sens))
        plt.title(ss)
        fig.savefig(op.join(fig_dir, ss + ".png"))
        del fig

    print("Generating html report...")
    html = '<h1>Models comparison</h1><div class="row"><ul>'
    for m in models:
        html += "<li>" + m + "/<li>"
    html += "</ul></div><div class=row>"
    for ss in ss_list:
        html += '<img src="' + op.join(fig_dir, ss + '.png') + '" height=200 width=auto/>'
    html += "</div>"

    with open(html_f, "w+") as f:
        f.write(html_intro() + html + html_outro())
    print('Done')

