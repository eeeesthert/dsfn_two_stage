#!/usr/bin/env python3
"""CLI for two-dimensional larynx ultrasound stitching."""
import argparse,logging,random
from pathlib import Path
import numpy as np
from src.io_utils import load_config
from src.pipeline import MODES,run_pipeline
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--target',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')));p.add_argument('--mode',choices=MODES,default='full');p.add_argument('--slic-k',type=int,choices=[50,100,150,200,300,400,500]);a=p.parse_args();random.seed(42);np.random.seed(42);logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(message)s');cfg=load_config(a.config)
 if a.slic_k:cfg['slic']['n_segments']=a.slic_k
 status=run_pipeline(a.reference,a.target,a.output,cfg,a.mode);return 0 if status['status']=='SUCCESS' else 2
if __name__=='__main__':raise SystemExit(main())
