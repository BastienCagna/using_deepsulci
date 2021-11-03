from dico_toolbox.database import FileDatabase
from using_deepsulci.cohort import Cohort
import os.path as op
import json


UDP_TEMPLATES = {
    "cohort": "cohorts/cohort-[cohort]_hemi-[hemi].json",
    "model": "models/cohort-[train_cohort]_hemi-[hemi]_model-[model]/run-[run]/cohort-[train_cohort]_hemi-[hemi]_model-[model]_[type].*",
    "evaluation_summary": "evaluations/cohort-[train_cohort]_hemi-[hemi]_model-[model]/run-[run]/cohort-[test_cohort]_hemi-[hemi].csv",
    "evaluation": "evaluations/cohort-[train_cohort]_hemi-[hemi]_model-[model]/run-[run]/[hemi][subject]_[session].[extension]"
}


class UDPDatabase(FileDatabase):
    def __init__(self, path=None, from_env=None) -> None:
        if path is None and from_env is None:
            raise ValueError("Either path or from_env must be passed.")
        if from_env:
            # Load environment file
            env = json.load(open(from_env))
            path = env["working_path"]
        super().__init__(path, templates=UDP_TEMPLATES)

    def get_from_template(self, template, **kwargs):
        return super().get_from_template(template, **kwargs)

    def get_cohort(self, name, hemi):
        cohort_f = self.get_from_template("cohort", cohort=name, hemi=hemi)[0]
        return Cohort(from_json=cohort_f)
