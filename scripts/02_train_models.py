"""
    This script train one or several model using labelled graphs of a given
    cohorts. The learning is very very long (like arround 36 hours for 140
    subjects using GPU).

    Example
    -------
    python 02_train_models.py -c pclean6_hemi-L -m unet3d --cuda 2 --lr 0.0025 --momentum 0.9 -s 1 2 3

python 02_train_models.py -c pclean50A_hemi-L -e env_active_learning.json --test pclean12A_hemi-L -m active_test --cuda 4 --dropout 0 --lr 0.0025 --momentum 0.9 --active --amount 3 --init 4 -r 1 --vs 4 --purge --strategy random
"""
import os.path as op
from deepsulci.sulci_labeling.capsul.training import SulciDeepTraining
from using_deepsulci.sulci_active_deep_training import SulciActiveDeepTraining
import using_deepsulci.similarity as sim
import json
from os import makedirs
from datetime import datetime
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt

from using_deepsulci.cohort import Cohort
from utils import Logger
import sys


def plot_sim_mat(mat, save_as):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(mat, interpolation="nearest", aspect="auto")
    plt.title("Graph Similarity Matrix")
    fig.savefig(save_as)


def train_cohort(cohort, out_dir, fname, dropout=0, lr=0, momentum=0,
                 batch_size=1, translation_file=None, steps=None, cuda=-1,
                 extend=None, active_mode=False, initial_amount=2,
                 inclusion_amount=1, max_iter=-1, retrain_from_scratch=False,
                 inclusion_strategy="proba_avg", test_cohort=None,
                 voxel_size=None, detailed_log=False):
    proc = SulciActiveDeepTraining() if active_mode else SulciDeepTraining()

    if extend:
        new_name = fname + "_ext-" + extend.name
        new_dir = op.realpath(op.join(out_dir, "..", new_name))
        makedirs(new_dir, exist_ok=True)
        print("Copying existing model files")
        for ext in ["_model.mdsm", "_params.json", "_traindata.json"]:
            shutil.copyfile(
                op.join(out_dir, fname + ext),
                op.join(new_dir, new_name + ext)
            )
        fname = new_name
        out_dir = new_dir

        cohort = cohort.concatenate(extend, cohort.name + '+' + extend.name)

    # Inputs
    # Active learning
    if active_mode:
        graphs = cohort.get_graphs()

        indexes = np.arange(len(graphs)-1)
        np.random.shuffle(indexes)
        init_graphs = []
        for idx in indexes[:initial_amount]:
            init_graphs.append(graphs[idx])
        for g in init_graphs:
            graphs.remove(g)

        if inclusion_strategy == "representativeness":
            sim_mat_f = op.join(out_dir, "graphs_similarity_matrix.npy")
            sim_mat = sim.graphs_similarity_matrix(graphs)
            np.save(sim_mat_f, sim_mat)
            plot_sim_mat(sim_mat, sim_mat_f)
            plt.show()
            proc.similarity_matrix = sim_mat_f

        proc.graphs = init_graphs
        proc.available_graphs = graphs
        proc.graphs_notcut = cohort.get_notcut_graphs()
        proc.inclusion_amount = inclusion_amount
        proc.inclusion_strategy = inclusion_strategy
        proc.test_graphs = test_cohort.get_graphs()
        proc.max_iter = max_iter
        proc.retrain_from_scratch = retrain_from_scratch
    else:
        proc.graphs = cohort.get_graphs()
        proc.graphs_notcut = cohort.get_notcut_graphs()
    proc.cuda = cuda
    proc.translation_file = translation_file

    # Steps
    proc.step_1 = not steps or (steps and 1 in steps)
    proc.step_2 = not steps or (steps and 2 in steps)
    proc.step_3 = not steps or (steps and 3 in steps)
    proc.step_4 = not steps or (steps and 4 in steps)
    # bool(len(cohort.get_notcut_graphs()))

    proc.voxel_size = voxel_size
    proc.batch_size = int(batch_size)
    proc.dropout = dropout
    proc.learning_rate = lr
    proc.momentum = momentum

    # Outputs
    proc.model_file = op.join(out_dir, fname + "_model.mdsm")
    proc.param_file = op.join(out_dir, fname + "_params.json")
    proc.log_file = op.join(out_dir, fname + "_log.csv")
    proc.detailed_log = detailed_log
    proc.traindata_file = op.join(out_dir, fname + "_traindata.json")
    if active_mode:
        proc.acc_log_file = op.join(out_dir, fname + "_acc_log.csv")

    # Run
    # if op.exists(proc.model_file) and not extend:
    #     print("Skipping the training. Model file already exist.")
    #     print(proc.model_file)
    # else:
    proc.run()


def main():
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('-c', dest='cohorts', type=str, nargs='+', default=None,
                        required=False, help='Cohort names')
    parser.add_argument('-x', dest='extends', type=str, default=None,
                        required=False,
                        help='Poursue training adding a new cohort')
    parser.add_argument('-s', dest='steps', type=int, nargs='+', default=None,
                        help='Steps to run')
    parser.add_argument('--active', dest='active', const=True, nargs='?',
                        default=False, help="Perform active learning.")
    parser.add_argument('--amount', dest='amount', type=int, default=1,
                        help='Number of graph to include at each iterations of '
                             'the active learning')
    parser.add_argument('--init', dest='init', type=int, default=2,
                        help='Number of graph used for the first iteration of '
                             'the active learning.')
    parser.add_argument('--strategy', dest='strategy', type=str,
                        default="proba_avg",
                        help='Inclusion strategy for active learning.')
    parser.add_argument('--max_iter', dest='max_iter', type=int, default=-1,
                        help='Maximum numver of active learning passes.')
    parser.add_argument('--test', dest='test', type=str, default=None,
                        required=False, help='Testing cohort name')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0,
                        help='Dropout')
    parser.add_argument('--batch', dest='batch', type=float, default=1,
                        help='Batch Size (default: 1)')
    parser.add_argument('--lr', dest='lr', type=float, default=0,
                        help='Learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0,
                        help='Momentum')
    parser.add_argument('-m', dest='modelname', type=str,  required=True,
                        help='Model name')
    parser.add_argument('--cuda', dest='cuda', type=int, default=-1,
                        help='Use a speciific cuda device ID or CPU (-1)')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    parser.add_argument('-r', dest='runs', type=int, default=1,
                        help="Number of runs to perform")
    parser.add_argument('--purge', dest='purge', const=True, nargs='?',
                        default=False, help="Delete previous results")
    parser.add_argument('--retrain', dest='retrain', const=True, nargs='?',
                        default=False,
                        help="Retrain from scratch for active learning")
    parser.add_argument('--pursue', dest='pursue', const=True, nargs='?',
                        default=False, help="Continue the training")
    parser.add_argument('--vs', dest='vs', type=float,
                        default=None, help='Target voxel size')
    parser.add_argument('--detail', dest='detail', const=True, nargs='?',
                        default=False, help="Detailed log")
    args = parser.parse_args()

    # Load environment file
    env_f = args.env if args.env else op.join(
        op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    cohorts_dir = op.join(env['working_path'], "cohorts")
    outdir = op.join(env['working_path'], "models")
    now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    makedirs(op.join(env["working_path"], "logs"), exist_ok=True)
    log_f = op.join(env["working_path"], "logs", "step_02_" + now + ".log")

    cohorts = []
    for c in args.cohorts:
        cfile = op.join(cohorts_dir, "cohort-" + c + ".json")
        cohorts.append(Cohort(from_json=cfile))
    cohorts = sorted(cohorts, key=len)

    if args.test:
        cfile = op.join(cohorts_dir, "cohort-" + args.test + ".json")
        test_cohort = Cohort(from_json=cfile)
    else:
        test_cohort = None

    if args.extends:
        extend = Cohort(from_json=op.join(
            cohorts_dir, "cohort-" + args.extends + ".json"))
        print("Extends with:", extend.name)
    else:
        extend = None

    for cohort in cohorts:
        print("\n\n ****** START TO TRAIN **********", cohort.name)
        for r in range(args.runs):
            print("\n***** RUN {}/{} ******".format(r+1, args.runs))
            fname = "cohort-" + cohort.name + "_model-" + args.modelname
            mdl_dir = op.join(outdir, fname, 'run-{:02d}'.format(r+1))

            if op.isdir(mdl_dir):
                if args.purge:
                    print(
                        "Erasing previous results for run-{:02d}".format(r+1))
                    shutil.rmtree(mdl_dir, ignore_errors=True)
                elif not args.pursue:
                    continue

            makedirs(mdl_dir, exist_ok=True)

            sys.stdout = Logger(log_f)
            train_cohort(cohort, mdl_dir, fname, args.dropout, args.lr,
                         args.momentum, args.batch, env['translation_file'],
                         args.steps, args.cuda, extend, args.active, args.init,
                         args.amount, args.max_iter, args.retrain,
                         args.strategy, test_cohort, args.vs, args.detail)
    return None


if __name__ == "__main__":
    main()
