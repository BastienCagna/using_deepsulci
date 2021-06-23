"""
    This script create JSON files that will be used to provide graphs when
    learning models. It defines a set of cohort based on several database where
    subjects' graphs have been mannually labeled.

    For now (08/04/2021), three databases are available with a total of 216
    subjects.

    This script need a env.json file saved in this directory that specify:
        - "bv_databases_path": Where all Brainvisa databased are located
        - "working_path": Where output files will be saved

    If any graph is missing, the whole database will be considered as
    unavailable.
"""

# Author : Bastien Cagna (bastiencagna@gmail.com)

import os.path as op
from os import makedirs
import json
import argparse
from using_deepsulci.cohort import bv_cohort, Cohort
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
        sgraph_f = op.join(subgraph_dir, op.split(sub.graph)[1][:-4] + '_filtered.arg')
        filter_sulci_names(sub.graph, sgraph_f, sulci_to_keep, default)
        cohort.subjects[i].graph = sgraph_f

        if sub.notcut_graph:
            sgraph_f = op.join(subgraph_dir, op.split(sub.notcut_graph)[1][:-4] + '_filtered.arg')
            filter_sulci_names(sub.notcut_graph, sgraph_f, sulci_to_keep, default)
            cohort.subjects[i].notcut_graph = sgraph_f
    return cohort


def foldico_cohorts(cohort_desc, hemi="both", composed_desc={}, modified_graphs_dir=None):
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
                        
            if modified_graphs_dir and 'keep_sulci_names' in desc.keys():
                d = op.join(modified_graphs_dir, cohort.name)
                cohort = filter_names_of_cohort(cohort, d, desc['keep_sulci_names'])

            if not do_not_add:
                cohorts[cname] = cohort
                all_cohortes.append(cohort)
                print("{}: {} subjects".format(cohort.name, len(cohort)))

    return all_cohortes


def main():
    parser = argparse.ArgumentParser(description='Create cohorts files (.json)')
    parser.add_argument('-e', dest='env', type=str, default=None, help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    cohorts_dir = op.join(env['working_path'], "cohorts")
    modg_dir = op.join(env['working_path'], "modified_graphs")
    makedirs(cohorts_dir, exist_ok=True)
    print("Cohorts will be saved to:", cohorts_dir)

    # Create all cohorts for both hemispheres
    cohorts = foldico_cohorts(env['cohorts'],
                              composed_desc=env['composed_cohorts'],
                              modified_graphs_dir=modg_dir)

    for cohort in cohorts:
        fname = "cohort-" + cohort.name + ".json"
        cohort.to_json(op.join(cohorts_dir, fname))


if __name__ == "__main__":
    main()
