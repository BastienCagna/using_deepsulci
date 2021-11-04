"""
    Basic learning study
    ====================

    Learn and test models on several cohorts sets.

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
    python 30_basic_learning.py -e env_basic_learning.json -n 24 -f --cuda 0 -vs 2
"""
# Author : Bastien Cagna (bastiencagna@gmail.com)
import argparse
import os.path as op
from utils import extend_folds, run


def learn_model(train_cohort, modelname, d, lr, m, r, c, env_file,
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
        c: int
            Cuda device index
    """
    cmd = 'cd ' + op.dirname(op.realpath(__file__)) + '; '
    cmd += "python 10_train_models.py -c {} -m {} --cuda {:d} " \
           "--dropout {:f} --lr {:f} --momentum {:f} -r {:d} -s 1 2 3 " \
           "-e {} --purge".format(
               train_cohort, modelname, c, d, lr, m, r, env_file
           )

    if voxel_size:
        cmd += ' --vs {}'.format(voxel_size)

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
    parser.add_argument('-n', dest='njobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('-f', dest='force', const=True, nargs='?', default=False,
                        help='Compute the new graph even if the  file already exist')
    parser.add_argument('--vs', dest='vs', type=float,  default=None,
                        help='Target voxel size')

    args = parser.parse_args()

    voxel_size = args['vs']

    # TODO: use a json to define this:
    # folds = [
    #     ('pclean12*', ['pclean50*', 'archi50*', 'hcp50*']),
    #     ('archi12*', ['pclean50*', 'archi50*', 'hcp50*']),
    #     ('hcp12*', ['pclean50*', 'archi50*', 'hcp50*']),
    #     ('pclean50*', ['pclean12*', 'archi12*', 'hcp12*']),
    #     ('archi50*', ['pclean12*', 'archi12*', 'hcp12*']),
    #     ('p25a25*', ['pclean12*', 'archi12*', 'hcp12*']),
    #     ('PClean', ['Archi', 'HCP']),
    #     ('HCP', ['Archi', 'PClean']),
    #     ('Archi',  ['PClean', 'HCP']),
    #     ('p54a70*', ['pclean08*', 'archi08*', 'hcp08*']),
    #     ('p54a70h68*', ['pclean08*', 'archi08*', 'hcp08*']),
    # ]
    # folds = extend_folds(folds)
    folds = [
        # ('pclean12A', ['pclean50A', 'archi50A', 'hcp50A']),
        ('pclean50A', ['pclean12A']),  # ['pclean12A', 'archi12A', 'hcp12A']),
        ('archi50A', ['pclean12A'])  # ['pclean12A', 'archi12A', 'hcp12A']),
        # ('p25a25A', ['pclean12A', 'archi12A', 'hcp12A']),
        # ('PClean', ['Archi', 'HCP']),
        # ('HCP', ['Archi', 'PClean']),
        # ('Archi', ['PClean', 'HCP']),
        # ('p54a70A', ['pclean08A', 'archi08A', 'hcp08A']),
        # ('p54a70h68A', ['pclean08A', 'archi08A', 'hcp08A']),
    ]

    print("{} models to learn".format(len(folds)))

    # Learn and test
    modelname = 'unet3d_d00b01'
    # for (train_cohort, test_cohorts) in folds:
    # for h in ['L', 'R']:
    #     t_cohort = train_cohort + '_hemi-' + h
    #     print("\n\n**** Learning on {}\n\n".format(train_cohort))
    #
    #     # Train
    #     learn_model(t_cohort, modelname, 0, .0025, .9, 1, int(args.cuda),
    #                 env_file, voxel_size)

    for (train_cohort, test_cohorts) in folds:
        for h in ['L']:  # , 'R']:
            t_cohort = train_cohort + '_hemi-' + h
            # Test
            for cohort in test_cohorts:
                cohort += '_hemi-' + h
                print("\n\n**** Learn on {} and testing on {}\n\n".format(
                    t_cohort, cohort
                ))
                m = 'cohort-' + t_cohort + '_model-' + modelname
                test_model(m, cohort, '1', args.njobs, args.env, voxel_size,
                           args.force)


if __name__ == "__main__":
    main()
