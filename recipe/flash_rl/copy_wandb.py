# following https://github.com/wandb/wandb/issues/5726

import pandas as pd
import wandb
import os

# export 
os.environ["WANDB_API_KEY"] = os.environ['WANDB_MSR_API_KEY']
os.environ["WANDB_BASE_URL"] = "https://microsoft-research.wandb.io"

wandb.login(key=os.environ['WANDB_MSR_API_KEY'], host="https://microsoft-research.wandb.io", force=True)

api = wandb.Api()

run_ids = [
    ... # add your run IDs here
]

runs = [api.run(idi) for idi in run_ids]
histories = [ri.history() for ri in runs]
files = [ri.files() for ri in runs]
configs = [ri.config for ri in runs]
names = [ri.name for ri in runs]


os.environ["WANDB_API_KEY"] = os.environ['WANDB_PUBLIC_API_KEY']
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"

wandb.login(key=os.environ['WANDB_PUBLIC_API_KEY'], host="https://api.wandb.ai", force=True)

for i in range(len(histories)):
    
    new_run = wandb.init(project='Flash-DAPO', entity='llychinalz', config=configs[i], name=names[i], resume="allow")
    for index, row in histories[i].iterrows():
        new_run.log(row.dropna().to_dict(), step=int(row['_step']))
    
    new_run.finish()
    