import os.path as op
from deepsulci.sulci_labeling.capsul import SulciDeepLabeling

TEST_DATA_ROOT = "/home/bastien/data/archi"
TEST_OUT = "/home/bastien/data/test_cnn_models/archi"


def test_labeling():
    # Config the labelling process
    ana_dir = op.join(TEST_DATA_ROOT,
                      "t1-1mm-1/001/t1mri/default_acquisition/default_analysis")
    mod_dir = op.join(TEST_DATA_ROOT, "models/custom_model/cnn_models")

    process = SulciDeepLabeling()
    # Input graph to segment
    process.graph = op.join(ana_dir, "folds/3.3/session1_manual",
                            "L001_session1_manual.arg")
    # Root file corresponding to the input graph
    process.roots = op.join(ana_dir, "segmentation/Lroots_001.nii.gz")
    # File (.mdsm) storing neural network parameters
    process.model_file = op.join(mod_dir, "sulci_unet_model_left.mdsm")
    # file (.json) storing the hyperparameters (cutting threshold)
    process.param_file = op.join(mod_dir, "sulci_unet_model_params_left.json")
    # ?
    process.rebuild_attributes = True
    # Skeleton file corresponding to the input graph
    process.skeleton = op.join(ana_dir, "segmentation/Lskeleton_001.nii.gz")
    process.allow_multithreading = True
    # Output labeled graph
    process.labeled_graph = op.join(TEST_OUT, "001_left.arg")
    # device on which to run the training (-1 for cpu, i>=0 for the i-th gpu)
    process.cuda = -1
    # Use same random sequence
    process.fix_random_seed = False

    # Run the labelling
    process._run_process()

    if not op.exists(process.labeled_graph):
        raise Exception("Labelled graph file does not exist.")

    # Compare mannually set labels and predicted ones
    


print(test_labeling())
