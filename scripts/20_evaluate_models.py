"""
    Evaluate a model on a test cohort
    =================================

    Labellize and compute ESI (local and global) for all subject of a cohort.

    Parameters
    ----------
    -c [cohort]: str
        Name of the cohort to use for the test.

    -e [path]: str (opt.)
        Set the path to the settings file (.json).
        If not specified, it use default settings.

    -m [model_name]: str
        Name of the model to evaluate.

    -r [1 2 ...]: [int] (opt.)
        List of runs to run.
        By default, only one learning run will be performed.

    -n [njobs]: int (opt.)
        Number of parallel jobs to use.
        Default is 1.

    -f: flag (opt.)
        Re-do the evaluation if the labelled graph already exist.

    Outputs
    -------
    Write labelled graph and corresponding scores .csv files in [working_path]/evaluations/[model]/run-[run_id]/.

    Examples
    --------
    python 20_evaluate_models.py -c pclean50A_hemi-L -m default_model -n 24 -f

"""
# Author : Bastien Cagna (bastiencagna@gmail.com)

from os import makedirs
from warnings import warn
import json
import os.path as op
import argparse
import pandas as pd
from using_deepsulci.settings import Settings, ResultDatabase
import numpy as np

from utils import html_intro, html_outro
from deepsulci.sulci_labeling.capsul.labeling import SulciDeepLabeling
# from deepsulci.sulci_labeling.capsul.error_computation import ErrorComputation

from using_deepsulci.cohort import Cohort
from using_deepsulci.processes.labeling_evaluation import LabelingEvaluation
from joblib import Parallel, delayed
from utils import real_njobs


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
    html += '<li>Accuracy: {:02f}% (+/- {:.02f}%)</li>'.format(100 *
                                                               avg_acc, 100*std_acc)
    html += '<li>Balanced accuracy: {:02f}% (+/- {:.02f}%)</li>'.format(
        100*avg_bacc, 100*std_bacc)
    html += '<li>Sensitivity: {:02f}% (+/- {:.02f}%)</li>'.format(
        100*avg_sens, 100*std_sens)
    html += '<li>Specificity: {:02f}% (+/- {:.02f}%)</li>'.format(
        100*avg_spec, 100*std_spec)
    html += '<li>ESI: {:02f}% (+/- {:.02f}%)</li>'.format(100 *
                                                          avg_esi, 100*std_esi)
    html += '<table class="table"><thead><tr><td>Sulci</td><td>Occurences</td><td>Acc.</td><td>B. Acc.</td><td>Sensitivity</td><td>Specificity</td><td>ESI</td></tr></thead><tbody>'
    for ss in ss_list:
        n = np.sum(df['s_' + ss] > 0)
        html += '<tr><td>{}</td><td>{}/{}</td><td>{:.01f}% (+/- {:.01f}%)</td><td>{:.01f}% ' \
                '(+/- {:.01f}%)</td><td>{:.01f}% (+/- {:.01f}%)</td><td>{:.01f}% (+/- {:.01f}%)</td><td>{:.01f}% (+/- {:.01f}%)</td>'.format(
                    ss, n, N, 100 *
                    np.mean(df['acc_' + ss]), 100*np.std(df['acc_' + ss]),
                    100*np.mean(df['bacc_' + ss]), 100 *
                    np.std(df['bacc_' + ss]),
                    100*np.mean(df['sens_' + ss]), 100 *
                    np.std(df['sens_' + ss]),
                    100*np.mean(df['spec_' + ss]), 100 *
                    np.std(df['spec_' + ss]),
                    100*np.mean(df['ESI_' + ss]), 100*np.std(df['ESI_' + ss])
                )
    html += '</tbody></table>'

    return html_intro() + html + html_outro()


def evaluation_job(db: ResultDatabase, cohort: Cohort, sub: str, model: str, run: str, ss_list: str,
                   voxel_size: str, n_iter=10, force=False):

    kwargs = {"train_cohort": cohort.name,
              "hemi": cohort.hemi, "model_name": model}
    model_file = db.generate_from_template(
        "model", run_id=run, **kwargs, makedirs=True)
    param_file = db.generate_from_template(
        "model_params", run_id=run, **kwargs, makedirs=True)

    labeleds = []
    labeled_graph = db.generate_from_template(
        "labelled_graph", subject=sub, **kwargs, makedirs=True)
    if force or not op.exists(labeled_graph):
        # Sulci Segmentation
        # seg_proc = ce.get_process_instance('morphologist.capsul.axon.sulcigraph')
        # # Inputs
        # seg_proc.skeleton = sub.skeleton
        # seg_proc.roots = sub.roots
        # seg_proc.grey_white = sub.grey_white
        # seg_proc.hemi_cortex = sub.hemi_cortex
        # seg_proc.split_brain = sub.split_brain
        # seg_proc.white_mesh = sub.white_mesh
        # seg_proc.pial_mesh = sub.pial_mesh
        # # Outputs
        # seg_proc.graph = labeled_graph
        # seg_proc.sulci_voronoi = labeled_graph[:-4] + '_voronoi.nii.gz'
        # seg_proc.run()

        # Graph labeling
        lab_proc = SulciDeepLabeling()
        lab_proc.graph = sub.graph
        lab_proc.roots = sub.roots
        lab_proc.skeleton = sub.skeleton
        lab_proc.model_file = model_file
        lab_proc.param_file = param_file
        lab_proc.rebuild_attributes = False
        lab_proc.labeled_graph = labeled_graph
        lab_proc.stat_file = labeled_graph[:-3] + 'json'
        lab_proc.distribution_iter = 0
        lab_proc.voxel_size = voxel_size
        lab_proc.verbose = False
        lab_proc.run()
    else:
        print(labeled_graph, "already exists")
        # labeleds.append(labeled_graph)

    # Labeling evaluation
    # esi_proc = ce.get_process_instance(
    #     'deepsulci.sulci_labeling.capsul.error_computation')
    scr_f = db.generate_from_template(
        "labelled_graph_stats", subject=sub, **kwargs, makedirs=True)
    if force or not op.exists(scr_f):
        esi_proc = LabelingEvaluation()
        esi_proc.t1mri = sub.t1
        esi_proc.true_graph = sub.graph
        esi_proc.labeled_graphs = labeleds
        esi_proc.sulci_side_list = ss_list
        esi_proc.scores_file = scr_f
        esi_proc.verbose = False
        esi_proc.run()
        print(sub.name, ' evaluated')
    else:
        print(scr_f, 'already exists')

    return scr_f


def evaluate_model(db: ResultDatabase, cohort_name: str, hemi: str, model: str, run: str,
                   voxel_size=None, n_jobs=1, force=False):
    print("Evaluate:", model)

    cohort = db.get_cohort(cohort_name, hemi)
    kwargs = {"train_cohort": cohort.name, "run": run,
              "hemi": cohort.hemi, "model_name": model}
    param_file = db.get_from_template("model_params", **kwargs)

    params = json.load(open(param_file))
    ss_list = params['sulci_side_list']

    if 'cutting_threshold' not in params.keys():
        # TODO: better manage of this or verify that the default value
        warn("/!\\ No cutting threshold, setting arbitrary value: 250")
        params['cutting_threshold'] = 250
        json.dump(params, open(param_file, 'w+'))

    # Labelize and get accuracy stats for each label
    n = real_njobs(n_jobs)
    print("Using {} parallel jobs for {} subjects".format(n, len(cohort.subjects)))
    scores_files = Parallel(n_jobs=n)(delayed(evaluation_job)
                                      (db, cohort, sub, model, run, ss_list, voxel_size,
                                       force) for sub in cohort.subjects)

    # Concatenate stats from all the cohort
    dframes = []
    for i, f in enumerate(scores_files):
        if f:
            dframes.append(pd.read_csv(f))
        else:
            print('Error for subject', cohort.subjects[i].name)
    all_scores = pd.concat(dframes)
    cohort_csv = db.generate_from_template(
        "labelled_cohort_stats", **kwargs, makedirs=True)
    all_scores.to_csv(cohort_csv)

    # HTML report
    cohort_report = db.generate_from_template(
        "labelled_cohort_report", **kwargs, makedirs=True)
    print("HTML report: ", cohort_report)
    with open(cohort_report, 'w+') as f:
        f.write(html_report(all_scores, ss_list))


def main():
    parser = argparse.ArgumentParser(description='Test trained CNN model')
    parser.add_argument('-c', dest='cohort', type=str, default=None, required=False,
                        help='Testing cohort name')
    parser.add_argument('-m', dest='model', type=str, default=None, required=False,
                        help='Model name')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    parser.add_argument('-r', dest='runs', type=int, nargs='+', default=[1],
                        help='Runs to process')
    parser.add_argument('--vs', dest='vs', type=float, default=None,
                        help='Target voxel size')
    parser.add_argument('-n', dest='njobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('-f', dest='force', const=True, nargs='?',
                        default=False, help='Compute the new graph even if the '
                                            'file already exist')
    args = parser.parse_args()

    # Load environnment file
    settings = Settings(args.env)

    for r in args.runs:
        evaluate_model(settings.outputs, args.cohort, args.hemi, args.model, "run-{:02}".format(r),
                       n_jobs=args.njobs, force=args.force, voxel_size=args.vs)


if __name__ == "__main__":
    main()
