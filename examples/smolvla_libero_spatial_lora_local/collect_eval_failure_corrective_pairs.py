#!/usr/bin/env python3
"""Collect expert corrective trajectories for exact failed evaluation initial states."""
from __future__ import annotations
import argparse,csv,json,os,subprocess
from pathlib import Path
import numpy as np
import validate_perturbation_replay as replay_tool
SCRIPT_DIR=Path(__file__).resolve().parent;REPO_ROOT=SCRIPT_DIR.parents[1]
PYTHON=Path('/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python')

def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation-root',type=Path,default=REPO_ROOT/'results/expanded_checkpoint_select/006000');p.add_argument('--record-dir',type=Path,default=SCRIPT_DIR/'work/eval_failure_corrective_pairs');p.add_argument('--output-dir',type=Path,default=REPO_ROOT/'results/eval_failure_corrective_pairs');p.add_argument('--max-donors',type=int,default=20);p.add_argument('--accepted-per-failure',type=int,default=1);p.add_argument('--collisions-only',action='store_true');p.add_argument('--tasks',nargs='*',default=None);p.add_argument('--confirm-local-generated-data',action='store_true');a=p.parse_args()
 if not a.confirm_local_generated_data:p.error('--confirm-local-generated-data is required')
 metadata={r['task_id']:r for r in csv.DictReader((a.evaluation_root/'tasks.csv').open())}
 failures=[]
 for result in sorted((a.evaluation_root/'raw').glob('batch_*/server_*.json')):
  for task in json.loads(result.read_text())['tracks'][0]['tasks']:
   for ep in task['episodes']:
    if not ep['success'] and (not a.collisions_only or bool(ep.get('collided'))):
     if a.tasks is None or task['task_name'] in set(a.tasks):failures.append((task['task_name'],int(ep['episode_id']),bool(ep.get('collided'))))
 tasks,episodes,frames=replay_tool.load_dataset_tables();task_index={str(name):int(row.task_index) for name,row in tasks.iterrows()}
 a.record_dir.mkdir(parents=True,exist_ok=True);a.output_dir.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['PYTHONPATH']=f"{REPO_ROOT/'LIBERO-plus'}:{REPO_ROOT}:{REPO_ROOT/'compe'}";env['LIBERO_ROOT']=str(REPO_ROOT/'LIBERO-plus');env['LD_LIBRARY_PATH']=':'.join(x for x in ('/home/cm-pc/miniconda3/envs/parc2026/lib',env.get('LD_LIBRARY_PATH','')) if x)
 results=[]
 for pair_index,(task_id,ordinal,base_collided) in enumerate(failures):
  row=metadata[task_id];base=row['base_task'].replace('_',' ')
  # Task table is language text; tolerate underscore-preserving scene names.
  candidates=[(name,index) for name,index in task_index.items() if name.replace(' ','_')==row['base_task'].replace(' ','_')]
  if not candidates:
   results.append({'task_id':task_id,'episode_id':ordinal,'status':'source_task_missing','base_task':row['base_task']});continue
  source_index=candidates[0][1];episode_ids=[]
  for _,ep in episodes.iterrows():
   eid=int(ep.episode_index)
   if source_index in set(map(int,frames.loc[frames.episode_index==eid,'task_index'].unique())):episode_ids.append(eid)
  if not episode_ids:
   results.append({'task_id':task_id,'episode_id':ordinal,'status':'source_episodes_missing'});continue
  donor_ordinals=[ordinal]+[x for x in range(len(episode_ids)) if x!=ordinal]
  attempts=[];accepted=[]
  for donor_ordinal in donor_ordinals[:a.max_donors]:
   source_episode=episode_ids[donor_ordinal];ef=frames[frames.episode_index==source_episode].sort_values('frame_index');actions=[np.asarray(x).tolist() for x in ef.action]
   tag=f"pair{pair_index:03d}_{task_id}__seed{ordinal}__donor{donor_ordinal:03d}";payload=a.output_dir/f'{tag}.payload.json';output=a.output_dir/f'{tag}.output.json';record_dir=a.record_dir/tag
   payload.write_text(json.dumps({'tasks_csv':str((a.evaluation_root/'tasks.csv').resolve()),'task_id':task_id,'init_ordinal':ordinal,'actions':actions,'record_path':None,'rl_record_dir':str(record_dir.resolve()),'camera_size':256}))
   if output.is_file():
    outcome=json.loads(output.read_text());resumed=True
   else:
    subprocess.run([str(PYTHON),str(Path(replay_tool.__file__).resolve()),'--worker-payload',str(payload),'--worker-output',str(output)],cwd=REPO_ROOT,env=env,check=True);outcome=json.loads(output.read_text());resumed=False
   attempt={'donor_ordinal':donor_ordinal,'source_episode':source_episode,**outcome};attempts.append(attempt);print(f"[{pair_index+1}/{len(failures)}] {task_id} seed={ordinal} donor={donor_ordinal}: success={outcome['accepted']} collision={outcome['collided']} resumed={resumed}",flush=True)
   if outcome['accepted'] and not outcome['collided']:
    accepted.append(attempt)
    # Add auditable pair provenance to the committed episode metadata.
    record=Path(outcome['rl_record_path'])
    with np.load(record,allow_pickle=False) as z:payload_np={k:np.asarray(z[k]) for k in z.files}
    m=json.loads(str(payload_np['metadata_json'].item()));m['corrective_pair']={'base_evaluation_root':str(a.evaluation_root.resolve()),'base_task_id':task_id,'base_episode_id':ordinal,'base_success':False,'base_collided':base_collided,'same_initial_state':True,'expert_source_episode':source_episode,'expert_donor_ordinal':donor_ordinal};payload_np['metadata_json']=np.asarray(json.dumps(m));tmp=record.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload_np);os.replace(tmp,record)
    if len(accepted)>=a.accepted_per_failure:break
  results.append({'task_id':task_id,'instruction':row['instruction'],'texture_suffix':row['texture_suffix'],'episode_id':ordinal,'base_collided':base_collided,'accepted':accepted,'attempts':attempts})
 report={'protocol':'exact_failed_eval_init_expert_corrective_search','evaluation_root':str(a.evaluation_root.resolve()),'collisions_only':a.collisions_only,'accepted_per_failure':a.accepted_per_failure,'n_base_failures':len(failures),'n_failures_with_corrective':sum(bool(x.get('accepted')) for x in results),'n_corrective_successes':sum(len(x.get('accepted',[])) for x in results),'results':results};(a.output_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(f"accepted {report['n_corrective_successes']} trajectories for {report['n_failures_with_corrective']}/{len(failures)} failures")
if __name__=='__main__':main()
