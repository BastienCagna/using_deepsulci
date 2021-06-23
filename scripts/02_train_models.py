"""
    This script train one or several model using labelled graphs of a given
    cohorts. The learning is very very long (like arround 36 hours for 140
    subjects using GPU).
"""
import os.path as op
from deepsulci.sulci_labeling.capsul.training import SulciDeepTraining
import json
from os import makedirs
from datetime import datetime
import argparse
import shutil

from using_deepsulci.cohort import Cohort
from using_deepsulci.utils.misc import add_to_text_file
from utils import Logger
import sys


def train_cohort(cohort, out_dir, fname, dropout=0, lr=0, momentum=0, translation_file=None, steps=None,
                 cuda=-1, extend=None):
    proc = SulciDeepTraining()

    if extend:
        new_name = fname + "_ext-" + extend.name
        print("Copying existing model files")
        for ext in ["_model.mdsm", "_params.json", "_traindata.json"]:
            shutil.copyfile(
                op.join(out_dir, fname + ext),
                op.join(out_dir, new_name + ext)
            )
        fname = new_name

        cohort = cohort.concatenate(extend, cohort.name + '+' + extend.name)

    # Inputs
    proc.graphs = cohort.get_graphs()
    proc.graphs_notcut = cohort.get_notcut_graphs()
    proc.cuda = cuda
    proc.translation_file = translation_file

    # Steps
    proc.step_1 = not steps or (steps and 1 in steps)
    proc.step_2 = not steps or (steps and 2 in steps)
    proc.step_3 = not steps or (steps and 3 in steps)
    proc.step_4 = not steps or (steps and 4 in steps)
    #bool(len(cohort.get_notcut_graphs()))

    proc.dropout = dropout
    proc.learning_rate = lr
    proc.momentum = momentum

    # Outputs
    proc.model_file = op.join(out_dir, fname + "_model.mdsm")
    proc.param_file = op.join(out_dir, fname + "_params.json")
    proc.log_file = op.join(out_dir, fname + "_log.csv")
    proc.traindata_file = op.join(out_dir, fname + "_traindata.json")

    # Run
    if op.exists(proc.model_file) and not extend:
        print("Skipping the training. Model file already exist.")
        print(proc.model_file)
    else:
        proc.run()


def main():
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('-c', dest='cohorts', type=str, nargs='+', default=None, required=False,
                        help='Cohort names')
    parser.add_argument('-x', dest='extends', type=str, default=None, required=False,
                        help='Poursue training adding a new cohort')

    parser.add_argument('-s', dest='steps', type=int, nargs='+', default=None,
                        help='Steps to run')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0,
                        help='Dropout')
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
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    cohorts_dir = op.join(env['working_path'], "cohorts")
    outdir = op.join(env['working_path'], "models")
    makedirs(outdir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    makedirs(op.join(env["working_path"], "logs"), exist_ok=True)
    log_f = op.join(env["working_path"], "logs", "step_02_" + now + ".log")

    cohorts = []
    for c in args.cohorts:
        cohorts.append(Cohort(from_json=op.join(cohorts_dir, "cohort-" + c + ".json")))
    cohorts = sorted(cohorts, key=len)

    if args.extends:
        extend = Cohort(from_json=op.join(cohorts_dir, "cohort-" + args.extends + ".json"))
    else:
        extend = None

    for cohort in cohorts:
        print("\n\n ****** START TO TRAIN ", cohort.name)
        fname = "cohort-" + cohort.name + "_model-" + args.modelname

        sys.stdout = Logger(log_f)
        train_cohort(cohort, outdir, fname, args.dropout, args.lr, args.momentum, env['translation_file'],
                     args.steps, args.cuda, extend)
    return None


if __name__ == "__main__":
    main()
