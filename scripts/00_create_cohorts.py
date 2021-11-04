"""
    Search files for all defined cohorts
    ====================================

    This script create JSON files that will be used to provide graphs when
    learning models. It defines a set of cohort based on several database where
    subjects' graphs have been mannually labeled.

    This script need a env.json file saved in this directory that specify:
        - "bv_databases_path": Where all Brainvisa databased are located
        - "working_path": Where output files will be saved

    /!\ If any graph is missing, the whole database will be considered as
    unavailable.

    Parameters
    ----------
    -e [path]: str (opt.)
        Set the path to the settings file (.json).
        If not specified, it use default settings.

    Outputs
    -------
    Write cohors JSON file in the [working_path]/cohorts directory.

    Examples
    --------
    python 00_create_cohorts.py -e env_active_learning.json
"""
# Author : Bastien Cagna (bastiencagna@gmail.com)

import os.path as op
from os import makedirs
import argparse
import numpy as np
import matplotlib.pyplot as plt
from using_deepsulci.cohort import bv_cohort, Cohort
from using_deepsulci.settings import Settings
from soma import aims


def filter_sulci_names(in_graph, out_graph, sulci_names_to_keep=[], default_value='unknown'):
    # graph conversion
    graph = aims.read(in_graph)

    for vertex in graph.vertices():
        if 'name' in vertex.keys() and vertex['name'] not in sulci_names_to_keep:
            vertex['name'] = default_value

        # for visualization purpose only
        if 'label' in vertex.keys():
            if 'name' in vertex.keys() and vertex['name'] in sulci_names_to_keep:
                print(vertex['name'], "have been found")
                vertex['label'] = vertex['name']
            else:
                vertex['label'] = default_value
    graph['label_property'] = 'label'

    # save graph
    aims.write(graph, out_graph)


def filter_names_of_cohort(cohort, subgraph_dir, sulci_to_keep=[], default='unknown'):
    makedirs(subgraph_dir, exist_ok=True)

    for i, sub in enumerate(cohort.subjects):
        sgraph_f = op.join(subgraph_dir, op.split(
            sub.graph)[1][:-4] + '_filtered.arg')
        filter_sulci_names(sub.graph, sgraph_f, sulci_to_keep, default)
        cohort.subjects[i].graph = sgraph_f

        if sub.notcut_graph:
            sgraph_f = op.join(subgraph_dir, op.split(
                sub.notcut_graph)[1][:-4] + '_filtered.arg')
            filter_sulci_names(sub.notcut_graph, sgraph_f,
                               sulci_to_keep, default)
            cohort.subjects[i].notcut_graph = sgraph_f
    return cohort


def foldico_cohorts(cohort_desc, hemi="both", composed_desc={}):
    """ Create all used cohorts based on several available databases.  """

    hemis = ["L", "R"] if hemi == "both" else [hemi]

    all_cohortes = []
    for h in hemis:
        cohorts = {}
        for cname, desc in cohort_desc.items():
            try:
                cohort = bv_cohort(cname, desc['path'], h,
                                   centers=desc["centers"],
                                   graph_v=desc['graph_v'],
                                   ngraph_v=desc['ngraph_v'],
                                   acquisition=desc['acquisition'],
                                   session=desc['session'],
                                   inclusion=desc['inclusion'],
                                   exclusion=desc['exclusion'],
                                   )
                cohorts[cname] = cohort
                all_cohortes.append(cohort)

                print("{}: {} subjects".format(cohort.name, len(cohort)))
            except IOError as exc:
                print(cname, "is unavailable")
                print("\tError: ", exc)

        for cname, desc in composed_desc.items():
            cohort = Cohort(cname + "_hemi-" + h, subjects=[])
            do_not_add = False
            for cname2 in desc['cohorts'].keys():
                if cname2 not in cohorts.keys():
                    print("{} is unavailable (need {})".format(cname,
                                                               cname2))
                    do_not_add = True
                    break

                if len(desc['cohorts'][cname2]["indexes"]) == 0:
                    cohort = cohort.concatenate(cohorts[cname2])
                else:
                    for subi in desc['cohorts'][cname2]["indexes"]:
                        if subi > len(cohorts[cname2]):
                            print("{} is unavailable (not enough subject in {})"
                                  .format(cname, cname2))
                            do_not_add = True
                            break
                        else:
                            cohort.subjects.append(
                                cohorts[cname2].subjects[subi])
                    if do_not_add:
                        break

            if not do_not_add:
                cohorts[cname] = cohort
                all_cohortes.append(cohort)
                print("{}: {} subjects".format(cohort.name, len(cohort)))

    return all_cohortes


def cohorts_plot(cohorts, hemi):
    n_rows = len(cohorts)
    subjects = []
    for c in cohorts:
        if c.name[0].isupper() and c.name.endswith(hemi):
            subjects.extend(set(s.name for s in c.subjects))
    n_cols = len(subjects)

    img = np.zeros((n_rows, n_cols))
    for ic, c in enumerate(cohorts):
        for j, sub in enumerate(subjects):
            if sub in c:
                img[ic, j] = 1

    fig = plt.figure(figsize=(12, 6))
    plt.imshow(img, interpolation="nearest", aspect="auto")
    plt.xticks(range(len(subjects)), subjects, rotation=60)
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create cohorts files (.json)')
    parser.add_argument('-e', dest='env', type=str,
                        default=None, help="Configuration file")
    args = parser.parse_args()

    # Load settings
    settings = Settings(args.env)

    # Create all cohorts for both hemispheres
    cohorts = foldico_cohorts(
        settings.get_parameter('cohorts'),
        composed_desc=settings.get_parameter('composed_cohorts', {}))

    # Save all the cohorts
    for cohort in cohorts:
        cohort.to_json(settings.outputs.generate_from_template(
            "cohort", name=cohort.name, hemi=cohort.hemi, makedirs=True))

    # Plot them
    for h in ['L', 'R']:
        fig = cohorts_plot(cohorts, h)
        fig.savefig(op.join())


if __name__ == "__main__":
    main()
