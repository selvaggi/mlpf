import uproot
import awkward as ak
import numpy as np
import plotly.graph_objects as go

# Open ROOT file
filename = "/eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/train/gun_dr_logE_211125_test/out_reco_edm4hep_1_0.parquet"
output = ak.from_parquet(filename)


# Symbol mapping
symbols = ["cross", "x", "circle", "square", "diamond"]

n_events = 100
print(f"Total events: {n_events}")

for i in range(1):
    print(f"\nShowing event {i+1}/{n_events} ... (press Enter for next, q to quit)")
    # Convert this event?s hits to numpy

    # Load jagged arrays (per event, do NOT flatten yet)
    X_hit = output["X_hit"][i]
    X_track = output["X_track"][i]

    hit_x = np.concatenate((np.array(X_hit[:,6]), np.array(X_track[:,12])), axis=0)
    hit_y = np.concatenate((np.array(X_hit[:,7]), np.array(X_track[:,13])), axis=0)
    hit_z = np.concatenate((np.array(X_hit[:,8]), np.array(X_track[:,14])), axis=0)
    hit_e = np.concatenate((np.array(X_hit[:,5]),np.array(X_track[:,5])), axis=0)# for tracks this is p
    hit_type_hit = np.array(X_hit[:,-2])+1
    hit_type_track = np.array(X_track[:,0])

    hit_type = np.concatenate((hit_type_hit,hit_type_track), axis=0)
 
    hit_genlink_hits =  np.array(output["ygen_hit"][i])
    hit_genlink_tracks =  np.array(output["ygen_track"][i])
    genlink = np.concatenate((hit_genlink_hits,hit_genlink_tracks), axis=0)

    x = hit_x
    y = hit_y
    z = hit_z
    e = hit_e
    t = hit_type
    g = np.array(genlink)

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
    print(hit_genlink_tracks)
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
