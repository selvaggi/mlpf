import wandb
import pandas as pd

# Authenticate with wandb
wandb.login()

# Define your entity and project name
entity = "fcc_ml"  # replace with your entity
project = "mlpf_debug"  # replace with your project name

# Initialize the wandb API
api = wandb.Api()

# Fetch the runs for the given project
runs = api.runs(f"{entity}/{project}")

# Extract required information
data = []
for run in runs:
    run_name = run.name
    run_url = run.url
    print(dir(run))
    1/0
    run_command = run.command #run.config.get('command', 'No command specified')  # Assumes 'command' is a config parameter

    data.append({
        "Run Name": run_name,
        "Run URL": run_url,
        "Run Command": run_command
    })

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Optionally, save to a CSV file
df.to_csv("/eos/home-g/gkrzmanc/wandb_runs.csv", index=False)
