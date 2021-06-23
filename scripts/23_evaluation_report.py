import pandas as pd
import numpy as np
import plotly.graph_objects as go


f = "/home/bastien/data/distant/deepsulci_models_comparison/evaluations/" \
    "cohort-archi6_hemi-L_model-unet3d/all_scores.csv"
df = pd.read_csv(f)

ss_list, bacc = [], []
for colname in df.columns:
    name = '_'.join(colname.split('_')[1:])
    if len(name) > 3:
        ss_list.append(name)
        bacc.append(np.mean(df['bacc_' + name]))

fig = go.Figure(
    data=[go.Bar(y=bacc)],
    layout_title_text="Average Balanced Accuracy"
)

# fig.update_layout(
#     xaxis = dict(
#         tickmode = 'array',
#         tickvals = np.arange(len(ss_list)),
#         ticktext = ss_list
#     )
# )
fig.show(renderer="svg")

# html = '<html><head>'
# html += '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">'
# html += '</head><body>'
# html += '<h1></h1>'
# html += '<table class="table"><thead><tr><td>Sulci</td><td>Acc.</td><td>B. Acc.</td><td>ESI</td></tr></thead><tbody>'
# for ss in ss_list:
#     html += '<tr><td>{}</td><td>{:.01f}% (+/- {:.01f}%)</td><td>{:.01f}% ' \
#             '(+/- {:.01f}%)</td><td>{:.01f}% (+/- {:.01f}%)</td>'.format(
#         ss, np.mean(df['acc_' + ss]), np.std(df['acc_' + ss]),
#         np.mean(df['bacc_' + ss]), np.std(df['bacc_' + ss]),
#         np.mean(df['ESI_' + ss]), np.std(df['ESI_' + ss])
#     )
# html += '</tbody></table>'
#
#
# html += '<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" ' \
#         'integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+' \
#         'IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>'
# html += '<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/' \
#         'js/bootstrap.bundle.min.js" integrity="sha384-ho+j7jyWK8fNQe+A12' \
#         'Hb8AhRq26LrZ/JpcUGGOn+Y7RsweNrtN/tE3MoK7ZeZDyx" crossorigin="ano' \
#         'nymous"></script>'
# html += '</body></html>'
#
# with open(f[:-4] + ".html", 'w+') as fp:
#     fp.write(html)
