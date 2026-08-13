#!/usr/bin/env python3
"""Train only on verified corrective successes and explicit zero-gate base successes."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from safetensors.torch import save_file
SCRIPT_DIR=Path(__file__).resolve().parent;REPO_ROOT=SCRIPT_DIR.parents[1]
sys.path.insert(0,str(REPO_ROOT/'submission_template'))
from residual_policy import GatedResidualPolicyConfig,VisualLanguageGatedResidualNetwork,language_features,visual_features

def is_corrective(path:Path)->bool:
    text=str(path).lower()
    return 'stove_corrective_records' in text and ('__donor_' in text or '__xy' in text)

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--base-model-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 records=[]
 for path in sorted(a.input_dir.rglob('*.npz')):
  with np.load(path,allow_pickle=False) as z:m=json.loads(str(z['metadata_json'].item()))
  if bool(m.get('success')):records.append((path,m,is_corrective(path)))
 positives=[r for r in records if r[2]];negatives=[r for r in records if not r[2]]
 if not positives or not negatives:raise RuntimeError(f'need positive and negative records: {len(positives)}, {len(negatives)}')
 states=[];bases=[];targets=[];visuals=[];languages=[];gates=[];classes=[]
 for path,m,positive in records:
  with np.load(path,allow_pickle=False) as z:
   state=np.asarray(z['observation.state'],np.float32);base=np.asarray(z['base_action'],np.float32);action=np.asarray(z['action'],np.float32);front=np.asarray(z['observation.front']);wrist=np.asarray(z['observation.wrist'])
  n=len(state);states.append(state);bases.append(base);targets.append(action[:,:6]-base[:,:6] if positive else np.zeros((n,6),np.float32));visuals.append(np.stack([visual_features(f,w) for f,w in zip(front,wrist)]));lang=language_features(m.get('instruction',''));languages.append(np.repeat(lang[None],n,0));gates.append(np.full((n,1),float(positive),np.float32));classes.append(np.full(n,int(positive),np.int64))
 state=np.concatenate(states);base=np.concatenate(bases);target=np.concatenate(targets);visual=np.concatenate(visuals);language=np.concatenate(languages);gate_target=np.concatenate(gates);positive_mask=np.concatenate(classes).astype(bool)
 sm,ss=state.mean(0),state.std(0).clip(1e-6);am,ass=base.mean(0),base.std(0).clip(1e-6)
 allowed=tuple(sorted({" ".join(m.get('instruction','').lower().split()) for _,m,_ in positives}))
 config=GatedResidualPolicyConfig(state_mean=tuple(map(float,sm)),state_std=tuple(map(float,ss)),base_action_mean=tuple(map(float,am)),base_action_std=tuple(map(float,ass)),allowed_instructions=allowed)
 inputs=np.concatenate([(state-sm)/ss,(base-am)/ass,visual,language],1).astype(np.float32);limits=np.asarray([config.translation_limit]*3+[config.rotation_limit]*3,np.float32);expected=np.clip(target/limits,-1,1).astype(np.float32)
 torch.manual_seed(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=VisualLanguageGatedResidualNetwork(config).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
 xt=torch.from_numpy(inputs);yt=torch.from_numpy(expected);gt=torch.from_numpy(gate_target);pm=torch.from_numpy(positive_mask);pos_idx=torch.where(pm)[0];neg_idx=torch.where(~pm)[0];gen=torch.Generator().manual_seed(a.seed)
 # Balanced batches prevent 84 zero-gate trajectories from overwhelming 7 corrective ones.
 half=a.batch_size//2
 for step in range(a.steps):
  idx=torch.cat([pos_idx[torch.randint(len(pos_idx),(half,),generator=gen)],neg_idx[torch.randint(len(neg_idx),(a.batch_size-half,),generator=gen)]])
  direction,gate=model(xt[idx].to(device));target_gate=gt[idx].to(device);prediction=direction*gate
  correction_loss=(prediction-yt[idx].to(device)).square().mean();gate_loss=torch.nn.functional.binary_cross_entropy(gate,target_gate);loss=correction_loss+0.5*gate_loss
  opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if step%500==0 or step+1==a.steps:print(f'step={step+1} loss={loss.item():.6f} correction={correction_loss.item():.6f} gate={gate_loss.item():.6f} pos_gate={gate[:half].mean().item():.4f} neg_gate={gate[half:].mean().item():.4f}',flush=True)
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n');save_file({k:v.detach().cpu() for k,v in model.state_dict().items()},str(a.output_dir/'model.safetensors'))
 manifest={'method':'paired_corrective_positive_base_success_zero_gate','positive_corrective_episodes':len(positives),'zero_gate_base_success_episodes':len(negatives),'positive_transitions':int(positive_mask.sum()),'zero_gate_transitions':int((~positive_mask).sum()),'steps':a.steps,'seed':a.seed,'base_model_dir':str(a.base_model_dir),'base_model_sha256':hashlib.sha256((a.base_model_dir/'model.safetensors').read_bytes()).hexdigest(),'positive_paths':[str(x[0]) for x in positives]}
 (a.output_dir/'training_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
