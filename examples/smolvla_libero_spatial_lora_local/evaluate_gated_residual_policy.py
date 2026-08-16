#!/usr/bin/env python3
"""Offline diagnostics for a visual-language gated residual."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
SCRIPT_DIR=Path(__file__).resolve().parent
REPO_ROOT=SCRIPT_DIR.parents[1]
sys.path.insert(0,str(REPO_ROOT/"submission_template"))
from residual_policy import GatedResidualPolicyConfig,VisualLanguageGatedResidualNetwork,language_features,visual_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,action='append',required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--output-scale',type=float);a=p.parse_args()
 config=GatedResidualPolicyConfig.from_json(a.model_dir/'config.json');model=VisualLanguageGatedResidualNetwork(config);model.load_state_dict(load_file(str(a.model_dir/'model.safetensors')));model.eval()
 output_scale=config.output_scale if a.output_scale is None else a.output_scale
 if not 0.0 <= output_scale <= 1.0: raise ValueError('--output-scale must be in [0, 1]')
 limits=np.asarray([config.translation_limit]*3+[config.rotation_limit]*3,dtype=np.float32)
 buckets=defaultdict(lambda:{'n':0,'base':0.,'residual':0.,'gate':0.})
 episodes=transitions=0
 for root in a.input_dir:
  for path in sorted(root.rglob('*.npz')):
   with np.load(path,allow_pickle=False) as z:
    m=json.loads(str(z['metadata_json'].item()));state=np.asarray(z['observation.state'],dtype=np.float32);base=np.asarray(z['base_action'],dtype=np.float32);action=np.asarray(z['action'],dtype=np.float32);front=np.asarray(z['observation.front']);wrist=np.asarray(z['observation.wrist'])
   lang=language_features(m.get('instruction',''),config.language_dim);visual=np.stack([visual_features(f,w) for f,w in zip(front,wrist)])
   x=np.concatenate([(state-np.asarray(config.state_mean))/np.asarray(config.state_std),(base-np.asarray(config.base_action_mean))/np.asarray(config.base_action_std),visual,np.repeat(lang[None],len(state),0)],1).astype(np.float32)
   with torch.inference_mode(): direction,gate=model(torch.from_numpy(x));correction=(direction*gate).numpy()*limits*output_scale
   pred=np.clip(base[:,:6]+correction,-1,1);be=np.square(base[:,:6]-action[:,:6]).mean(1);re=np.square(pred-action[:,:6]).mean(1)
   keys=['all','success' if m.get('success') else 'failure',m.get('instruction','unknown')]
   for key in keys:
    b=buckets[key];b['n']+=len(state);b['base']+=float(be.sum());b['residual']+=float(re.sum());b['gate']+=float(gate.sum())
   episodes+=1;transitions+=len(state)
 report={'episodes':episodes,'transitions':transitions,'output_scale':output_scale,'buckets':{k:{'transitions':v['n'],'base_mse':v['base']/v['n'],'residual_mse':v['residual']/v['n'],'relative_mse_change':v['residual']/v['base']-1,'mean_gate':v['gate']/v['n']} for k,v in buckets.items()}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
