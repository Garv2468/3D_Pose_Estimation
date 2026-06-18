import pandas as pd
import json

def export_json(df: pd.DataFrame, path):
    return df.to_json(path, orient='records', indent=4)

def export_summary(metadata, path):
    with open(path, 'w') as f:
        f.write(json.dumps(metadata, indent=4))