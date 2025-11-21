import uproot
import awkward as ak
import numpy as np
import plotly.graph_objects as go

# Open ROOT file
filename = "output/290825/pf_tree_1.root"
tree = uproot.open(filename)["events"]

# Load jagged arrays (per event, do NOT flatten yet)
hit_x = tree["hit_x"].array()
hit_y = tree["hit_y"].array()
hit_z = tree["hit_z"].array()
hit_e = tree["hit_e"].array()
hit_type = tree["hit_type"].array()
hit_genlink0 = tree["hit_genlink0"].array()
hit_genlink1 = tree["hit_genlink3"].array()

# Symbol mapping
symbols = ["cross", "x", "circle", "square", "diamond"]

n_events = len(hit_x)
print(f"Total events: {n_events}")

for i in range(n_events):
    print(f"\nShowing event {i+1}/{n_events} ... (press Enter for next, q to quit)")

    # Convert this event?s hits to numpy
    x = np.array(hit_x[i])
    y = np.array(hit_y[i])
    z = np.array(hit_z[i])
    e = np.array(hit_e[i])
    t = np.array(hit_type[i])
    g = np.array(hit_genlink1[i])

    if len(x) == 0:
        print("No hits in this event.")
        inp = input()
        if inp.lower() == "q":
            break
        continue

    cmin = g.min()
    cmax = g.max()

    # Marker size scaling
    # marker_sizes = 5 + 10 * (e / e.max())
    marker_sizes = 10 * e/e

    # Build figure with one trace per hit_type
    fig = go.Figure()

    for tval in np.unique(t):
        mask = t == tval
        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode="markers",
                name=f"hit_type {tval}",
                marker=dict(
                    size=marker_sizes[mask],
                    color=g[mask],
                    colorscale="Turbo",
                    cmin=cmin,
                    cmax=cmax,
                    opacity=0.8,
                    symbol=symbols[int(tval) % len(symbols)],
                    colorbar=dict(title="genlink0"),
                    showscale=bool(tval == np.unique(t)[0])
                ),
                text=[
                    f"type={tt}, e={ee:.2f}, gen={gg}"
                    for tt, ee, gg in zip(t[mask], e[mask]*1e3, g[mask])
                ],
                hoverinfo="text"
            )
        )

    # Layout
    fig.update_layout(
            scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                        xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
                        yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
                        zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
                    ),
            paper_bgcolor="black",  # sets the area around the plot
            plot_bgcolor="black",   # sets the area behind the axes
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(title="Hit Type"),
            title=f"Event {i}",
        )

    # Save to HTML file (overwrites each time for the same event)
    html_filename = f"event_{i+1:04d}.html"
    fig.write_html(html_filename)

    # Open in browser
    fig.show()

    # Wait for user input before next event
    inp = input()
    if inp.lower() == "q":
        break
