import os.path as op
from using_deepsulci.cohort import Cohort
from using_deepsulci.processes.labels_list import SulciList
import argparse
from os import makedirs
import json
import numpy as np
import pandas as pd


def html_report(cohort, counts):
    ss_list = list(counts.keys())
    ss_list.remove('graph')
    ss_list.remove('Unnamed: 0')
    ss_list = sorted(ss_list)

    presents = []
    absents = []
    for ss in ss_list:
        presents.append(np.sum(counts[ss] > 0))
        absents.append(np.sum(counts[ss] == 0))
    presents, absents = np.array(presents), np.array(absents)
    n_graphs = len(counts['graph'])

    html = '<html><head><title>Consistency of ' + cohort.name + '</title>'
    html += '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">'
    html += '</head><body>'
    html += '<h1>Labels occurences</h1>'
    html += '<p>{} labels</p><p>{} are presents in all graph</p>'.format(len(ss_list), np.sum(presents == n_graphs))
    html += '<table class="table"><tr><th>Label</th><th>Present in:</th><th>Absent in:</th></tr>'
    for i, ss in enumerate(ss_list):
        html += '<tr><td>{}</td><td>{}</td><td>{}</td>'.format(ss, presents[i], absents[i])
    html += '</table>'
    return html + '</body></html>'


def cohort_sulci_list(cohort, outf):
    process = SulciList()
    process.graphs = list(sub.graph for sub in cohort.subjects)
    process.notcut = list(sub.notcut_graph for sub in cohort.subjects)
    process.report = outf
    process.run()

    with open(outf[:-3] + 'html', 'w+') as f:
        f.write(html_report(cohort, pd.read_csv(outf)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resample cohort files to desired resolution')
    parser.add_argument('-c', dest='cohorts', type=str, nargs='+', default=None, required=False,
                        help='Cohort names')
    # parser.add_argument('-n', dest='njobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('-e', dest='env', type=str, default=None, help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    check_dir = op.join(env['working_path'], "consistency_check")
    makedirs(check_dir, exist_ok=True)

    # Set cohorts paths
    chrt_dir = op.join(env['working_path'], "cohorts")
    cohorts_files = []
    for cname in args.cohorts:
        cohort_sulci_list(
            Cohort(from_json=op.join(chrt_dir, "cohort-" + cname + ".json")),
            op.join(check_dir, "cohort-" + cname + ".csv")
        )
