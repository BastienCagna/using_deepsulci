import os.path as op
from deepsulci.sulci_labeling.capsul.training import SulciDeepTraining

TEST_DATA_ROOT = "/home/bastien/data/archi"
TEST_OUT = "/home/bastien/data/test_deepsulci/cnn_training"


def test_cnn_training():
    process = SulciDeepTraining()

    graphs = [
        '/home/bastien/data/archi/t1-1mm-1/001/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/session1_manual/L001_session1_manual.arg',
        '/home/bastien/data/archi/t1-1mm-1/002/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/session1_manual/L002_session1_manual.arg',
        '/home/bastien/data/archi/t1-1mm-1/005/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/session1_manual/L005_session1_manual.arg',
        '/home/bastien/data/archi/t1-1mm-1/011/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/session1_manual/L011_session1_manual.arg'
    ]
    notcut = [
        '/home/bastien/data/archi/t1-1mm-1/001/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/L001.arg',
        '/home/bastien/data/archi/t1-1mm-1/002/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/L002.arg',
        '/home/bastien/data/archi/t1-1mm-1/005/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/L005.arg',
        '/home/bastien/data/archi/t1-1mm-1/011/t1mri/default_acquisition/'
        'default_analysis/folds/3.3/L011.arg'
    ]
    # Training base graphs
    process.graphs = graphs
    # Training base graphs before manual cutting of the elementary folds
    process.graphs_notcut = notcut
    # Device on which to run the training (-1 for cpu, i>=0 for the i-th gpu)
    process.cuda = 0
    # (optional) File (.trl) containing the translation of the sulci to
    # applied on the training base graphs (optional)
    #process.translation_file = None
    # Perform the data extraction step from the graphs
    process.step_1 = True
    # Perform the hyperparameter tuning step (learning rate and momentum)
    process.step_2 = True
    # Perform the model training step
    process.step_3 = True
    # Perform the cutting hyperparameter tuning step
    process.step_4 = True

    # File (.mdsm) storing neural network parameters
    process.model_file = op.join(TEST_OUT, "cnn_model.mdsm")
    # File (.json) storing the hyperparameters (learning rate, momentum,
    # cutting threshold)
    process.param_file = op.join(TEST_OUT, "cnn_model_params.json")
    # File (.json) storing the data extracted from the training base graphs
    process.traindata_file = op.join(TEST_OUT, "cnn_model_traindata.json")

    process._run_process()
    
    
test_cnn_training()
