from using_deepsulci.cohort import Cohort
import os.path as op
from os import makedirs
import json
from soma import aims
from deepsulci.deeptools.dataset import extract_data


def html_report(data, filepath):
    html = "<h1>Cohortes</h1>"

    all_labels = set()
    for cname in data.keys():
        for label in data[cname]:
            all_labels.add(label)

    html += "<table><tr><th></th>"
    for cname in data.keys():
        html += "<th>" + cname + "</th>"
    html += "</tr>"

    for label in all_labels:
        html += "<tr><td>" + label + "</td>"
        for cname in data.keys():
            s = "X" if label in data[cname] else ""
            html += "<td>" + s + "</td>"
        html += "</tr>"
    html += "</table>"

    f = open(filepath, 'w')
    f.write(html)
    f.close()


if __name__ == "__main__":
    env = json.load(open(op.join(op.split(__file__)[0], "env.json")))
    cohorts_dir = op.join(env['working_path'], "cohorts")
    report_dir = op.join(env['working_path'], "reports", "cohortes")

    makedirs(report_dir, exist_ok=True)

    data = {}
    for h in ["L", "R"]:
        cohorts = []
        for cname in env['cohortes']:
            cfile = "cohort-" + cname + '_hemi-' + h + ".json"
            cohorts.append(Cohort(from_json=op.join(cohorts_dir, cfile)))
        print(len(cohorts), " cohortes")

        for c in cohorts:
            print("\t{}: {} subjects".format(c.name, len(c)))
            sulci_side_list = {}
            for s in cohorts.subjects:
                graph = aims.read(s.graph)
                data = extract_data(graph)
                for n in data['names']:
                    if n in sulci_side_list:
                        sulci_side_list[n].append(s.name)
                    else:
                        sulci_side_list[n] = [s.name]
            data[c.name + "_hemi-" + h] = sulci_side_list

    html_report(data)