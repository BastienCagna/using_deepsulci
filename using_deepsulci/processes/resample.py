from soma import aimsalgo, aims
import os.path as op
import sigraph

# f = "/host/home/bastien/data/deepsulci_learning/nmr/sujet01/t1mri/t1/default_analysis/nobias_sujet01.nii.gz"
#
# job = aimsalgo.CubicResampler(aims.read(f))
# print(job)


graph_file = "/home/bastien/R113821.arg"
# graph_file = "/home/bastien/data/deepsulci_learning/nmr/sujet01/t1mri/t1/default_analysis/folds/3.3/base2018_manual/Lsujet01_base2018_manual.arg"

trl = "/casa/host/build/share/brainvisa-share-5.0/nomenclature/translation/sulci_model_2008.trl"

flt = sigraph.FoldLabelsTranslator()
flt.readLabels(trl)

graph = aims.read(graph_file)
flt.translate(graph)
for vertex in graph.vertices():
    vname = vertex.get('name')
    for item in vertex.get("aims_ss"):
        print(item)
    print(vname)
    exit()
