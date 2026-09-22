"""Read local W&B binary histories without network, GPU use, or changes to runs."""
import argparse,csv,json,statistics
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2


def save(path,obj):
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def scan(path,history=False):
 ds=DataStore();ds.open_for_scan(str(path));run=None;errors=[];end=path.stat().st_size;points=[]
 try:
  while ds._fp.tell()<end:
   try:b=ds.scan_data()
   except Exception as e:errors.append(str(e));break
   if b is None:break
   r=wandb_internal_pb2.Record();r.ParseFromString(b)
   if r.HasField('run'):
    run=dict(run=r.run.run_id,name=r.run.display_name,path=str(path),config={x.key:json.loads(x.value_json) for x in r.run.config.update})
    if not history:break
   if history and r.HasField('history'):
    data={x.key or '.'.join(x.nested_key):json.loads(x.value_json) for x in r.history.item}
    data['training_step']=r.history.step.num
    keep={k:v for k,v in data.items() if k.startswith('val/test_score/') or k in ['training_step','_step','_timestamp','critic/score/mean','timing_s/step','opd/evidence_residual_alpha','opd/effective_distillation_coef']}
    if len(keep)>1:points.append(keep)
 finally:ds._fp.close()
 return run,points,errors


def get(c,key,default=None):
 for k in key.split('.'):
  if not isinstance(c,dict) or k not in c:return default
  c=c[k]
 return c


def main():
 p=argparse.ArgumentParser();p.add_argument('--wandb',default='wandb');p.add_argument('--out',default='reports/er_baseline_curves_20260915');a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);inventory=[];histories={};errors=[]
 for path in sorted(Path(a.wandb).glob('run-*/*.wandb')):
  try:r,_,err=scan(path)
  except Exception as e:errors.append(dict(path=str(path),error=str(e)));continue
  if not r:continue
  c=r['config'];model=get(c,'actor_rollout_ref.model.path','');name=r['name']
  if not any(x in (model+' '+name).lower() for x in ['0.5b','05b','0_5b']):continue
  # Keep duplicate run IDs separate, since resumed logging directories may overlap.
  r['local_id']=path.parent.name;inventory.append(r)
  _,points,err=scan(path,True);histories[r['local_id']]=points
  if err:errors.append(dict(path=str(path),error=err))
  vals=[p for p in points if any(k.startswith('val/test_score/') for k in p)]
  print(r['run'],name,'train',max([x['training_step'] for x in points],default=None),'val',[(x['training_step'],round(x.get('val/test_score/Avg',-1)*100,2),round(x.get('val/test_score/hotpotqa',-1)*100,2)) for x in vals],flush=True)
 save(out/'inventory.json',inventory);save(out/'histories.json',histories);save(out/'read_errors.json',errors)
 rows=[]
 for r in inventory:
  c=r['config'];vals=[p for p in histories[r['local_id']] if 'val/test_score/Avg' in p]
  for v in vals:
   row=dict(local_id=r['local_id'],run=r['run'],name=r['name'],student=get(c,'actor_rollout_ref.model.path'),
     target=get(c,'algorithm.opd.teacher_target'),alpha=get(c,'algorithm.opd.evidence_residual_alpha'),
     distill=get(c,'algorithm.opd.lambda_distill'),grpo=get(c,'algorithm.opd.grpo_reward_coef'),
     n_agent=get(c,'actor_rollout_ref.rollout.n_agent'),train_source=get(c,'data.train_data_source'),
     val_file=get(c,'data.val_files'),**v)
   rows.append(row)
 if rows:
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with (out/'validation_points.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 print('Saved',len(inventory),'runs;',len(rows),'validation records to',out)

if __name__=='__main__':main()
