import json
import argparse
import pandas as pd
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PPO log files.")
    parser.add_argument(
        "-i", "--input_file", type=str, required=True, help="Path to the input log file."
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=True, help="Path to the output log file."
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        lines = f.readlines()

    df = pd.DataFrame()
    li = None
    for line in lines:
        if line.strip():
            line = line.replace("'", '"')
            di = json.loads(line)
            if li is None:
                match = re.search(r'global_step_(\d+)', di['model'])
                assert match
                li = {'step': int( match.group(1))}
            else:
                li['score'] = di['test_score/openai/gsm8k']
                df = pd.concat([df, pd.DataFrame([li])], ignore_index=True)
                li = None 
                
    df = df.sort_values(by='step')
    
    df.to_csv(args.output_file)