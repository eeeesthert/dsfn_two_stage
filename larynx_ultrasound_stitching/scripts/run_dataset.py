"""Run independent pairs without aborting the batch on a failed case."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.io_utils import load_config
from src.pipeline import run_pipeline
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,help='CSV columns: case,reference,target');p.add_argument('--output',required=True);p.add_argument('--config',default=str(Path(__file__).parents[1]/'config.yaml'));p.add_argument('--mode',default='full');a=p.parse_args();cfg=load_config(a.config);results=[]
with open(a.manifest,newline='') as f:
 for row in csv.DictReader(f):
  try: result=run_pipeline(row['reference'],row['target'],Path(a.output)/row['case'],cfg,a.mode)
  except Exception as e: result={'status':'FAILED_H','message':str(e)}
  results.append({'case':row['case'],**result})
Path(a.output).mkdir(parents=True,exist_ok=True);(Path(a.output)/'batch_status.json').write_text(json.dumps(results,indent=2))
