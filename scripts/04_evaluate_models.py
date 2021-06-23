import sys
import json
import os.path as op
from os import makedirs
import argparse
import pandas as pd
import signal
import os
import numpy as np

from utils import html_intro, html_outro
from deepsulci.sulci_labeling.capsul.labeling import SulciDeepLabeling
# from deepsulci.sulci_labeling.capsul.error_computation import ErrorComputation

from using_deepsulci.cohort import Cohort
from using_deepsulci.processes.labeling_evaluation import LabelingEvaluation
from joblib import Parallel, delayed
from utils import real_njobs

from capsul.api import capsul_engine


def html_report(df, ss_list):
    N = len(df["s_"+ss_list[0]])
    acc, bacc, sens, spec, esi = [], [], [], [], []
    for ss in ss_list:
        acc.extend(np.nan_to_num(df['acc_' + ss]))
        bacc.extend(np.nan_to_num(df['bacc_' + ss]))
        sens.extend(np.nan_to_num(df['sens_' + ss]))
        spec.extend(np.nan_to_num(df['spec_' + ss]))
        esi.extend(np.nan_to_num(df['ESI_' + ss]))
    avg_acc, std_acc = np.mean(acc), np.std(acc)
    avg_bacc, std_bacc = np.mean(bacc), np.std(bacc)
    avg_sens, std_sens = np.mean(sens), np.std(sens)
    avg_spec, std_spec = np.mean(spec), np.std(spec)
    avg_esi, std_esi = np.mean(esi), np.std(esi)

    html = '<h1></h1>'
    html += '<p>Averaged results:</p><ul>'
    html += '<li>Accuracy: {:02f}% (+/- {:.02f}%)</li>'.format(100*avg_acc, 100*std_acc)
    html += '<li>Balanced accuracy: {:02f}% (+/- {:.02f}%)</li>'.format(100*avg_bacc, 100*std_bacc)
    html += '<li>Sensitivity: {:02f}% (+/- {:.02f}%)</li>'.format(100*avg_sens, 100*std_sens)
    html += '<li>Specificity: {:02f}% (+/- {:.02f}%)</li>'.format(100*avg_spec, 100*std_spec)
    html += '<li>ESI: {:02f}% (+/- {:.02f}%)</li>'.format(100*avg_esi, 100*std_esi)
    html += '<table class="table"><thead><tr><td>Sulci</td><td>Occurences</td><td>Acc.</td><td>B. Acc.</td><td>Sensitivity</td><td>Specificity</td><td>ESI</td></tr></thead><tbody>'
    for ss in ss_list:
        n = np.sum(df['s_' + ss] > 0)
        html += '<tr><td>{}</td><td>{}/{}</td><td>{:.01f}% (+/- {:.01f}%)</td><td>{:.01f}% ' \
                '(+/- {:.01f}%)</td><td>{:.01f}% (+/- {:.01f}%)</td><td>{:.01f}% (+/- {:.01f}%)</td><td>{:.01f}% (+/- {:.01f}%)</td>'.format(
            ss, n, N, 100*np.mean(df['acc_' + ss]), 100*np.std(df['acc_' + ss]),
            100*np.mean(df['bacc_' + ss]), 100*np.std(df['bacc_' + ss]),
            100*np.mean(df['sens_' + ss]), 100*np.std(df['sens_' + ss]),
            100*np.mean(df['spec_' + ss]), 100*np.std(df['spec_' + ss]),
            100*np.mean(df['ESI_' + ss]), 100*np.std(df['ESI_' + ss])
        )
    html += '</tbody></table>'

    return html_intro() + html + html_outro()


def evaluation_job(sub, labeled_dir, model_file, param_file, ss_list, esi_dir):
    g_fname = op.split(sub.graph)[1]
    labeled_graph = op.join(labeled_dir, g_fname)

    if not op.exists(labeled_graph):
        lab_proc = SulciDeepLabeling()
        lab_proc.graph = sub.graph
        lab_proc.roots = sub.roots
        lab_proc.skeleton = sub.skeleton
        lab_proc.model_file = model_file
        lab_proc.param_file = param_file
        lab_proc.rebuild_attributes = False
        lab_proc.labeled_graph = labeled_graph
        lab_proc.run()
    else:
        print(labeled_graph, "already exists")

    # esi_proc = ce.get_process_instance(
    #     'deepsulci.sulci_labeling.capsul.error_computation')
    scr_f = op.join(esi_dir, g_fname[:-4] + '_scores.csv')
    if not op.exists(scr_f):
        esi_proc = LabelingEvaluation()
        esi_proc.t1mri = sub.t1
        esi_proc.true_graph = sub.graph
        esi_proc.labeled_graphs = [labeled_graph]
        esi_proc.sulci_side_list = ss_list
        esi_proc.scores_file = scr_f
        esi_proc.run()
    else:
        print(scr_f, 'already exists')
    return scr_f


def evaluate_model(cohort, model_file, param_file, labeled_dir, esi_dir=None,
                   n_jobs=1):
    # ce = capsul_engine()
    esi_dir = labeled_dir if esi_dir is None else esi_dir
    params = json.load(open(param_file))

    ss_list = params['sulci_side_list']

    if 'cutting_threshold' not in params.keys():
        # TODO: better manage of this or verify that the default value
        print("/!\\ No cutting threshold, setting arbitrary value: 250")
        params['cutting_threshold'] = 250
        json.dump(params, open(param_file, 'w+'))

    # results = parallel()
    scores_files = Parallel(n_jobs=real_njobs(n_jobs))(delayed(evaluation_job) \
      (sub, labeled_dir, model_file, param_file, ss_list, esi_dir)
        for sub in cohort.subjects)

    dframes = []
    for i, f in enumerate(scores_files):
        if f:
            dframes.append(pd.read_csv(f))
        else:
            print('Error for subject', cohort.subjects[i].name)
    all_scores = pd.concat(dframes)

    all_scores.to_csv(op.join(esi_dir, "cohort-" + cohort.name + ".csv"))
    print("HTML report: ", op.join(esi_dir, "cohort-" + cohort.name + ".html"))
    with open(op.join(esi_dir, "cohort-" + cohort.name + ".html"), 'w+') as f:
        f.write(html_report(all_scores, ss_list))


def main():
    parser = argparse.ArgumentParser(description='Test trained CNN model')
    parser.add_argument('-c', dest='cohort', type=str, default=None, required=False,
                        help='Testing cohort name')
    parser.add_argument('-m', dest='model', type=str, default=None, required=False,
                        help='Model name')
    # parser.add_argument('--cuda', dest='cuda', type=int, default=-1,
    #                     help='Use a speciific cuda device ID or CPU (-1)')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    parser.add_argument('-n', dest='njobs', type=int, default=1, help='Number of parallel jobs')
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))
    model_dir = op.join(env['working_path'], "models")
    cohort_dir = op.join(env['working_path'], "cohorts")

    print("Evaluate:", args.model)

    # cohortname = modelname.split("_model")[0]
    cohort_f = op.join(cohort_dir, args.cohort + ".json")

    model_f = op.join(model_dir, args.model + "_model.mdsm")
    params_f = op.join(model_dir, args.model + "_params.json")

    out_d = op.join(env['working_path'], "evaluations", args.model)
    makedirs(out_d, exist_ok=True)
    # fname = modelname + "_teston-" + cohortname + ".npy"
    # evaluate_model(Cohort(from_json=cohort_f), env['translation_file'],
    #                model_f, params_f, op.join(out_d, fname))
    evaluate_model(Cohort(from_json=cohort_f), model_f, params_f,
                   labeled_dir=out_d, n_jobs=args.njobs)


if __name__ == "__main__":
    main()
