import os.path as op
from joblib import cpu_count
from using_deepsulci.cohort import Cohort
from datetime import datetime
import sys


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message, flush=False):
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.terminal.write(message)
        if flush:
            self.flush()
        self.log.write(now + "\t" + message)

    def flush(self):
        self.terminal.flush()


def real_njobs(n):
    return min(n, cpu_count()) if n > 0 else cpu_count() - n


def read_cohorts(cpath, cnames):
    cnames = cnames if isinstance(cnames, list) else [cnames]

    cohorts = []
    for cname in cnames:
        cohorts.append(Cohort(from_json=op.join(cpath, "cohort-" + cname + ".json")))
    return cohorts


def html_intro(title=""):
    html = '<html><head>'
    html += '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">'
    html += '</head><body>'
    return html

def html_outro():
    html = '<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" ' \
            'integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+' \
            'IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>'
    html += '<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/' \
            'js/bootstrap.bundle.min.js" integrity="sha384-ho+j7jyWK8fNQe+A12' \
            'Hb8AhRq26LrZ/JpcUGGOn+Y7RsweNrtN/tE3MoK7ZeZDyx" crossorigin="ano' \
            'nymous"></script>'
    return html + '</body></html>'