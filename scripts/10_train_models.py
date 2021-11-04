"""
    A command line interface for learning processes
    ===============================================

    This script train one or several model using labelled graphs of a given cohorts.
    The learning is long (like arround 36 hours for 140 subjects using a GPU).

    Parameters
    ----------

    -c [cohort_1 cohort_2...]: [str]
        List of cohorts names

    -s [1 2 ...]: [int]
        Steps (list of int) to process:
            - 1: load graph data
            - 2: search hyperparameters
            - 3: train the final model
            - 4: recut the graphs based on the model predictions

    -m [name]: str
        Model name used to create ouputs files.

    Optional parameters for both modes
    ----------------------------------
    --batch [batch]: int (opt.)
        Set the batch size for the learning process.
        Default is 1.

    --cuda [device]: int (opt.)
        Specify the cuda device to use. If -1, it use the CPU.
        Default is -1.

    --detail: flat (opt.)
        Save a detailed log that register more information about the learning at each step.

    -e [path]: str (opt.)
        Set the path to the settings file (.json).
        If not specified, it use default settings.

    --lr [lr]: float (opt.)
        If not equal to 0, it fixes the learning rate (will not be estimated in step 2).
        Default is 0.

    --momentum [momentum]: float (opt.)
        If not equal to 0, it fixes the momentum (will not be estimated in step 2).
        Default is 0.

    --pursue: flag
        Continue the training of a previously trained model.

    -r [1 2 ...]: [int] (opt.)
        List of runs to run.
        By default, only one learning run will be performed.

    --vs [voxel_size]: float (opt.)
        Isotropic voxel size used to resample data before the learning.
        By default, no resampling will be performed.

    -x: str (opt.)
        Extend the cohorte name when pursuiing learning of a previously learning model with a new cohort.


    Parameters for active learning
    ------------------------------
    --active: flag
        When set, the active learning mode is activated.

    --test [cohort]: str
        For active learning mode. Cohorte that should be used to test the learn model at each iteration.

    --amount [amount]: int (opt.)
        For active learning mode. Set the active batch size (number of graph to add at each iteration).
        Default is 1.

    --init [init]: int (opt.)
        For active learning mode. Set the number of graphs used by the first iteration. 
        Default is 2.

    --strategy [strategy]: str (opt.)
        For active learning mode. Set the inclusion strategy that is used at each iteration to select the graphs 
        that need to be added for the next iteration.
        Available strategies are:
            - "random": samples are selected randomly with a uniform probability.
        Default is "random".

    --max_iter [max_iter]: int (opt.)
        For active learning mode. Maximum number of iteration. If equal to -1, active learning iterates until all the graphs have been used.
        Default is -1.

    --retrain: flag (opt.)
        For active learning mode. Retrain the model from scratch at each itertion.
        By default, the model will not be retrain from zero.

    Outputs
    -------
    Write the extracted data, final model, model parameters and logs in [working_path]/models/[model_path]/run-[run_id]/.

    Examples
    --------
    Basic training:
    python 02_train_models.py -c pclean6_hemi-L -m unet3d --cuda 2 --lr 0.0025 --momentum 0.9 -s 1 2 3

    Active learning:
    python 02_train_models.py -c pclean50A_hemi-L -e env_active_learning.json --test pclean12A_hemi-L -m active_test --cuda 4 --dropout 0 --lr 0.0025 --momentum 0.9 --active --amount 3 --init 4 -r 1 --vs 4 --purge --strategy random
"""
# Author : Bastien Cagna (bastiencagna@gmail.com)

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
from using_deepsulci.settings import Settings
from utils import Logger
import sys


def plot_sim_mat(mat, save_as):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(mat, interpolation="nearest", aspect="auto")
    plt.title("Graph Similarity Matrix")
    fig.savefig(save_as)


def train_cohort(cohort, settings: Settings, run: str, lr=0, momentum=0,
                 batch_size=1, translation_file=None, steps=None, cuda=-1,
                 active_mode=False, initial_amount=2,
                 inclusion_amount=1, max_iter=-1, retrain_from_scratch=False,
                 inclusion_strategy="proba_avg", test_cohort=None,
                 voxel_size=None, detailed_log=False):
    proc = SulciActiveDeepTraining() if active_mode else SulciDeepTraining()

    out = settings.outputs

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
            sim_mat_f = out.generate_from_template(
                "similarity_matrix", cohort_name=cohort.name, hemi=cohort.hemi)
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

    proc.voxel_size = voxel_size
    proc.batch_size = int(batch_size)
    proc.dropout = 0
    proc.learning_rate = lr
    proc.momentum = momentum

    # Outputs
    kwargs = {"train_cohort": cohort.name,
              "hemi": cohort.hemi, "run_id": run}
    proc.model_file = out.generate_from_template("model",  **kwargs)
    proc.param_file = out.generate_from_template("model_params", **kwargs)
    proc.log_file = out.generate_from_template("model_log", **kwargs)
    proc.detailed_log = detailed_log
    proc.traindata_file = out.generate_from_template("model_data", **kwargs)
    if active_mode:
        proc.acc_log_file = out.generate_from_template(
            "model_active_log", **kwargs)

    # Run
    proc.run()


def main():
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('-c', dest='cohorts', type=str, nargs='+', default=None,
                        required=False, help='Cohort names')
    parser.add_argument('-h', dest='hemi', type=str, default='X',
                        required=False, help='Hemisphere (L/R/X)')
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
                        default="random",
                        help='Inclusion strategy for active learning.')
    parser.add_argument('--max_iter', dest='max_iter', type=int, default=-1,
                        help='Maximum numver of active learning passes.')
    parser.add_argument('--test', dest='test', type=str, default=None,
                        required=False, help='Testing cohort name')
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

    # Load settings
    settings = Settings(args.env)
    log_f = settings.outputs.new_log_file("training", args.model_name)

    # outdir = settings.outputs.generate_from_template("model",op.join(env['working_path'], "models")

    # Load cohorts
    cohorts = sorted(settings.outputs.get_cohort(c, args.hemi) for c in args.cohorts, key=len)
    test_cohort = settings.outputs.get_cohort(
        args.test, args.hemi) if args.test else None

    for cohort in cohorts:
        print("\n\n ****** START TO TRAIN **********", cohort.name)
        for r in range(args.runs):
            print("\n***** RUN {}/{} ******".format(r+1, args.runs))

            run = "{:02d}".format(r)
            model_files = settings.outputs.get_from_template(
                "model", train_cohort=cohort.name, model_name=args.modelname, run=run)
            if len(model_files) > 0:
                mdl_dir, _ = op.split(model_files[0])
                if args.purge:
                    print(
                        "Erasing previous results for run-{:02d}".format(r+1))
                    shutil.rmtree(mdl_dir, ignore_errors=True)
                elif not args.pursue:
                    continue

            sys.stdout = Logger(log_f)
            train_cohort(cohort, settings, run, args.lr,
                         args.momentum, args.batch, settings.get_parameter(
                             'translation_file', None),
                         args.steps, args.cuda, args.active, args.init,
                         args.amount, args.max_iter, args.retrain,
                         args.strategy, test_cohort, args.vs, args.detail)
    return None


if __name__ == "__main__":
    main()
