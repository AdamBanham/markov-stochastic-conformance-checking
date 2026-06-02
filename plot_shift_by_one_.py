import plotly.express as px
import json
import pandas as pd
from os.path import join
from glob import glob

dump_directory = join(".", "evaluation", "shift-by-one")


stat_files = glob(join(dump_directory, "*_scores_v2.json"))

for stats in stat_files:
    json_data = json.load(open(stats, "r"))
    figure_name = stats.split("shift-by-one\\")[1].split("_scores")[0]

    # make stat data
    row_data = []
    for measure in json_data.keys():
        measure_data = json_data[measure]

        for window in measure_data:
            window_data = measure_data[window]
            for i, sample in enumerate(window_data):
                row_data.append((sample, measure, window, int(i * 4)))

    df = pd.DataFrame(row_data, columns=["score", "measurement", "window", "sample"])

    # make plot
    fig = px.scatter(
        df,
        x="sample",
        y="score",
        color="measurement",
        facet_col="measurement",
        symbol="window",
        marginal_y="histogram",
        labels=dict(score="measurement", measurement="technique",
                    sample="% of log swapped out"),
        width=1000,
        height=275,
        opacity=0.66
    )
    fig.update_xaxes(
        minallowed=-2, maxallowed=102, gridcolor='LightGray',
        zeroline=True, zerolinecolor='LightGray'
    )
    fig.update_yaxes(
        minallowed=0, maxallowed=1.05,
        range=(0, 1.05),
        gridcolor='LightGray',
        zeroline=True, zerolinecolor='LightGray'
    )
    fig.update_layout({
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'legend': {'font': {'size': 8}},
        'margin': {'t': 24, 'b': 24, 'l': 20, 'r': 20, 'pad': 8},
    })

    # save out image
    fig.write_image(join(dump_directory, f"{figure_name}_plot.pdf"))
