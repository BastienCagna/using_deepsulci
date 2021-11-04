from os import makedirs
from warnings import warn
import json
import os.path as op
from datetime import datetime
from dico_toolbox.database import FileDatabase
from dico_toolbox import test_data
from using_deepsulci.cohort import Cohort


TEMPLATES = {
    "cohort": "cohortes/cohort-[name]_hemi-[hemi].csv",
    "similarity_matrix": "similarity_matrix/cohort-[cohort_name]_hemi-[hemi]_similarity.npy",
    "model": "models/cohort-[train_cohort]_model-[model_name]/run-[run]/cohort-[train_cohort]_model-[model_name].mdsm",
    "model_params": "models/cohort-[train_cohort]_model-[model_name]/run-[run]/cohort-[train_cohort]_model-[model_name]_params.json",
    "model_data": "models/cohort-[train_cohort]_model-[model_name]/run-[run]/cohort-[train_cohort]_model-[model_name]_traindata.json",
    "model_log": "models/cohort-[train_cohort]_model-[model_name]/run-[run]/cohort-[train_cohort]_model-[model_name]_log.csv",
    "model_active_log": "models/cohort-[train_cohort]_model-[model_name]/run-[run]/cohort-[train_cohort]_model-[model_name]_active_log.csv",
    "labelled_graph": "evaluations/cohort-[train_cohort]_model-[model_name]/run-[run]/[subject]_[labelling_session][hemi].arg",
    "labelled_graph_stats": "evaluations/cohort-[train_cohort]_model-[model_name]/run-[run]/[subject]_[labelling_session][hemi]_scores.csv",
    "labelled_cohort_stats": "evaluations/cohort-[train_cohort]_model-[model_name]/run-[run]/cohort-[test_cohort]_hemi-[hemi]_scores.csv",
    "labelled_cohort_report": "evaluations/cohort-[train_cohort]_model-[model_name]/run-[run]/cohort-[test_cohort]_hemi-[hemi]_scores.html",
    "processing_log": "logs/[process]/[id]_[datetime].log"
}


class ResultDatabase(FileDatabase):
    def __init__(self, path: str):
        super().__init__(
            path,
            directory_levels=["output_type"],
            templates=TEMPLATES,
            allowed_extensions='*',
            forbidden_extensions=[]
        )

    def get_cohort(self, name: str, hemi: str) -> Cohort:
        return Cohort(from_json=self.get_from_template("cohort", name=name, hemi=hemi))

    def new_log_file(self, process, id):
        now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        return self.generate_paths_from_templates("processing_log", process=process, id=id, datetime=now)[0]


class Settings:
    def __init__(self, env_file=None, priority_to_env_file=False, priority_warnings=True, **kwargs) -> None:
        """ Read parameters and create output database.

        Args:
            env_file (str, optional): Path to settings file (.json). Default to None.
            priority_to_env_file (bool, optional): If True, parameters that are in settings file and kwargs get value from env_file. Defaults to False.
            priority_warnings (bool, optional): Warn when parameters are defined in settings file and kwargs. Defaults to True.
        """
        self.env_file = env_file
        self.params = kwargs
        if self.env_file:
            file_params = json.load(open(env_file, 'r'))
            for k in file_params.keys():
                if k in self.params:
                    if priority_to_env_file:
                        if priority_warnings:
                            warn("Overwritting parameter '{}' to match settings file".format(
                                k, file_params[k]))
                        self.params[k] = file_params[k]
                    elif priority_warnings:
                        warn("Parameter '{}' is also defined in the settings files but priority "
                             "is given to function arguments (set priotity_to_env_file to True to "
                             "overwritte with values from settings file".format(k))

        if not 'bv_databases' in self.params or len(self.params['bv_databases']) == 0:
            bv_db = test_data.create_test_database()
            self.params['bv_databases'] = {
                "dico_toolbox_test": bv_db.path
            }
            warn(
                "Using Dico Toolbox test database as no BrainVISA database is specified in settings.")

        self.outputs = ResultDatabase(self.get_parameter('working_path'))

    def get_parameter(self, key: str, default="----undefined---") -> any:
        if not key in self.params:
            if default == "----undefined---":
                raise ValueError(
                    "Undefined parameter '{}' in {}".format(key, self.env_file))
            return default
        return self.params[key]

    def get_path(self, template_name: str, **kwargs) -> str:
        if not template_name in self.outputs_db.templates:
            raise ValueError("Undefined path for '{}'.".format(template_name))
        return self.outputs.get_from_template(template_name, **kwargs)
