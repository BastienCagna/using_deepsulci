"""
    Active learning study
    =====================

    Parameters
    ----------
    --cuda [device]: int (opt.)
        Specify the cuda device to use. If -1, it use the CPU.
        Default is -1.

    -e [path]: str (opt.)
        Set the path to the environment file (.json).
        If not specified, it use the env.json file that should be created in the scripts/ folder.

    -n [njobs]: int (opt.)
        Number of parallel jobs to use.
        Default is 1.

    -f: flag (opt.)
        Re-do the evaluation if the labelled graph already exist.

    --vs [voxel_size]: float (opt.)
        Isotropic voxel size used to resample data before the learning.
        By default, no resampling will be performed.

    Outputs
    -------


    Examples
    --------
    python 31_active_learning.py -e env_active_learning.json -n 24 -f --cuda 0 -vs 2
"""
# Author : Bastien Cagna (bastiencagna@gmail.com)

import argparse
import os.path as op
from utils import extend_folds, run


def learn_active_model(train_cohort, modelname, d, lr, m, r, init, amount,
                       strat, maxIt, test_cohort, retrain, c, env_file,
                       voxel_size=None):
    """ Run CNN learning script on a cohort

        Parameters
        ==========
        train_cohort: str
            Name of the cohort.
        modelname: str
            Name of the model.
        d: float
            Dropout amount (between 0 and 1)
        lr: float
            Learing rate
        m: float
            Momentum
        r: int
            Number of runs
        init: int
            Number of subjects used for the first pass
        amount: int
            Number of added subjects for each pass
        strat: str
            One of this inclusion strategy:
            * 'random'
        maxIt: int
            Maximal number of learning pass
        test: str
            Name of the global evaluation cohort
        retrain: bool
            Retrain from scratch at each new pass
        c: int
            Cuda device index
        voxel_size: float
            Voxel size used for the inputs
        env_file: str
            Path to the JSON configuration file
    """
    cmd = 'cd ' + op.dirname(op.realpath(__file__)) + '; '
    cmd += "python 02_train_models.py -c {} -m {} --cuda {:d} " \
           "--dropout {:f} --lr {:f} --momentum {:f} -r {:d} -s 1 2 3 " \
           "-e {} --active --init {:d} --amount {:d} --strategy {} " \
           "--max_iter {:d} --test {} --purge".format(
               train_cohort, modelname, c, d, lr, m, r, env_file, init, amount, strat,
               maxIt, test_cohort
           )

    if voxel_size:
        cmd += ' --vs {}'.format(voxel_size)

    if retrain:
        cmd += ' --retrain'

    run(cmd)


def test_model(modelname, test_cohort, r, n_jobs, env_file, voxel_size=None,
               force=False):
    cmd = 'cd ' + op.dirname(op.realpath(__file__)) + '; '
    cmd += "python 04_evaluate_models.py -c cohort-{} -m {} -r {} -n {:d} " \
           "-e {}".format(test_cohort, modelname, r, n_jobs, env_file)

    if voxel_size:
        cmd += ' --vs {}'.format(voxel_size)
    if force:
        cmd += ' -f'
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('--cuda', dest='cuda', type=int, default=-1,
                        help='Use a specific cuda device ID or CPU (-1)')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    parser.add_argument('-n', dest='njobs', type=int, default=24,
                        help='Number of parallel jobs')
    parser.add_argument('-f', dest='force', const=True, nargs='?',
                        default=False, help='Compute the new graph even if the '
                                            'file already exist')
    args = parser.parse_args()

    # TODO: use .json file to speficy cohortes and parameters

    # Learn and test
    voxel_size = 2
    h = 'L'
    init = 10
    amount = 5
    # train, test = 'p54a70h68A_hemi-' + h, 'p08a08h08A_hemi-' + h
    # train, test = 'p12a12h12A_hemi-' + h, 'p04a04h04A_hemi-' + h
    # train, test = 'p04a04h04A_hemi-' + h, 'p01a01h01A_hemi-' + h
    # strat, sfx = 'random', 'r'
    # strat, sfx = 'intersub_proba_dist', 'ipd'

    configs = [
        ('pclean50A_hemi-' + h, 'pclean12A_hemi-' + h, 'random', 'r'),
        # ('pclean50A_hemi-' + h, 'pclean12A_hemi-' + h, 'median_certainty', 'mc')
    ]
    # configs = [
    #     ('p54a70h68A_hemi-' + h, 'p08a08h08A_hemi-' + h, 'random', 'r'),
    #     ('p54a70h68A_hemi-' + h, 'p08a08h08A_hemi-' + h, 'intersub_proba_dist','ipd')
    # ]

    for i, (train, test, strat, sfx) in enumerate(configs):
        print("TEST {}: {} {} {} {}".format(i, train, test, strat, sfx))
        modelname = 'unet3d_d00b01_active_i{:03}a{:03d}{}'.format(
            init, amount, sfx)
        learn_active_model(
            train,
            modelname,
            0,
            0.025,
            0.9,
            3,
            init,
            amount,
            strat=strat,
            maxIt=-1,
            retrain=False,
            test_cohort=test,
            c=args.cuda,
            env_file=args.env,
            voxel_size=voxel_size
        )


if __name__ == "__main__":
    main()
