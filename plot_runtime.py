import plotly.express as px
import json
import pandas as pd
from os.path import join
from glob import glob

dump_directory = join(".", "evaluation", "shift-by-one")


stat_files = glob(join(dump_directory, "*_runtimes.json"))
row_data = []

for stats in stat_files:
    json_data = json.load(open(stats, "r"))
    log_name = stats.split("shift-by-one\\")[1].split("_runtimes")[0]

    print(log_name)
    if "road_traffic" in log_name:
        log_name = "Road Fines"
    elif "sepsis" in log_name:
        log_name = "Sepsis"
    elif "travel" in log_name:
        log_name = "BPIC 2020 (Travel)"
    print(log_name)

    # make stat data
    for measure in ["precision", "recall"]:
        measure_data = json_data[measure]
        measure_key = f'compute_stochastic_entropy_{measure}'

        for window in ["1w", "2w", "3w"]:
            if window not in measure_data:
                continue
            window_data = measure_data[window]

            for segment in window_data:
                if segment == "_value":
                    continue

                segment_data = window_data[segment]['_value']
                measurement_data = segment_data[measure_key]
                net_data = segment_data['intersect_nets']

                for runtime, net_size in zip(measurement_data, net_data):
                    row_data.append((runtime, measure, window, net_size, log_name))

df = pd.DataFrame(row_data, columns=["runtime", "measurement", "window", "net_size", "log_name"])

# make plot
fig = px.scatter(
    df,
    x="net_size",
    y="runtime",
    color="log_name",
    facet_col="measurement",
    symbol="window",
    marginal_y="histogram",
    labels=dict(runtime="Runtime in Seconds", measurement="technique",
                net_size="# nodes in intersection", log_name="Event Log"),
    width=1000,
    height=375,
    opacity=0.33,
    log_y=True
)
fig.update_xaxes(
    minallowed=0, gridcolor='LightGray',
    zeroline=True, zerolinecolor='LightGray'
)
fig.update_yaxes(
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
fig.write_image(join(dump_directory, f"evaluation_runtime_plot.pdf"))
