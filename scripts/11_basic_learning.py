
import argparse
import os.path as op
from utils import extend_folds, run


def learn_model(train_cohort, modelname, d, lr, m, r, c, env_file):
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
    cmd += "python 02_train_models.py -c {} -m {} --cuda {:d} " \
           "--dropout {:f} --lr {:f} --momentum {:f} -r {:d} -s 1 2 3 " \
           "--resampling .5 -e {}".format(
        train_cohort, modelname, c, d, lr, m, r, env_file
    )

    run(cmd)


def test_model(modelname, test_cohort, r, n_jobs, env_file):
    cmd = 'cd ' + op.dirname(op.realpath(__file__)) + '; '
    cmd += "python 04_evaluate_models.py -c cohort-{} -m {} -r {} -n {:d} -e {}".\
        format(test_cohort, modelname, r, n_jobs, env_file)
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('--cuda', dest='cuda', type=int, default=-1,
                        help='Use a specific cuda device ID or CPU (-1)')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    args = parser.parse_args()

    env_file = args.env if args.env else \
        op.join(op.dirname(op.realpath(__file__)), 'env.json')

    folds = [
        ('pclean12*', ['p25a25*']),
        ('archi12*', ['p25a25*']),
        ('pclean50*', ['pclean12*', 'archi12*']),
        ('archi50*', ['pclean12*', 'archi12*']),
        # ('p25a25*', ['pclean12*', 'archi12*']),
        # ('PClean', ['Archi']),
        # ('Archi',  ['PClean']),
        # ('p54a70*', ['pclean08*', 'archi08*']),
    ]
    folds = extend_folds(folds)

    print("{} models to learn".format(len(folds)))

    # Learn and test
    modelname = 'unet3d_d00b01'
    # for (train_cohort, test_cohorts) in folds:
    #     for h in ['L', 'R']:
    #         t_cohort = train_cohort + '_hemi-' + h
    #         print("\n\n**** Learning on {}\n\n".format(train_cohort))
    #
    #         # Train
    #         learn_model(t_cohort, modelname, 0, .0025, .9, 3, int(args.cuda),
    #                     env_file)

    for (train_cohort, test_cohorts) in folds:
        for h in ['L', 'R']:
            t_cohort = train_cohort + '_hemi-' + h
            # Test
            for cohort in test_cohorts:
                cohort += '_hemi-' + h
                print("\n\n**** Learn on {} and testing on {}\n\n".format(
                    t_cohort, cohort
                ))
                m = 'cohort-' + t_cohort + '_model-' + modelname
                test_model(m, cohort, '1 2 3', 24, env_file)


if __name__ == "__main__":
    main()
