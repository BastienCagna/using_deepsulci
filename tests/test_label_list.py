import os.path as op
from ..labels_list import SulciList

TEST_DATA_ROOT = "/home/bastien/data/archi"
TEST_OUT = "/home/bastien/data/test_cnn_models/archi"


def test_labeling():
    process = SulciList()

    process.graphs = [
        '/home/bastien/data/archi/t1-1mm-1/001/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/session1_manual/L001_session1_manual.arg',
        '/home/bastien/data/archi/t1-1mm-1/002/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/session1_manual/L002_session1_manual.arg',
        '/home/bastien/data/archi/t1-1mm-1/005/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/session1_manual/L005_session1_manual.arg',
        '/home/bastien/data/archi/t1-1mm-1/011/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/session1_manual/L011_session1_manual.arg'
    ]
    process.notcut = [
        '/home/bastien/data/archi/t1-1mm-1/001/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/L001.arg',
        '/home/bastien/data/archi/t1-1mm-1/002/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/L002.arg',
        '/home/bastien/data/archi/t1-1mm-1/005/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/L005.arg',
        '/home/bastien/data/archi/t1-1mm-1/011/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/L011.arg'
    ]

    process._run_process()


test_labeling()
