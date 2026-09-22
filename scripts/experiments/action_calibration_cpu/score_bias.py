"""Complete canonical action-path scores on common frozen student prefixes."""
import argparse
import gc
import time
from common import ROOT,OUT,setup,load_model,verify_inputs,path_score,rows,append,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['calibration','test']);args=parser.parse_args()
    torch=setup();m=verify_inputs()
    if args.split=='test':assert (OUT/'bias.frozen.json').exists(),'Calibrate and freeze before test scoring'
    path=OUT/f'scores_{args.split}.jsonl';existing=rows(path)
    done={(r['label'],r['qid'],r['condition']) for r in existing};assert len(done)==len(existing)
    cases=[c for c in m['cases'] if c['split']==args.split];total=len(cases)*len(m['checkpoints'])
    for checkpoint in m['checkpoints']:
        pending=[c for c in cases if (checkpoint['label'],c['qid'],c['condition']) not in done]
        if not pending:continue
        print('LOAD_STUDENT',args.split,checkpoint['label'],flush=True)
        model=load_model(ROOT/checkpoint['path'])
        for c in pending:
            tic=time.time()
            scores={a:path_score(model,c['ids'],tag,m['vocab_size'],torch) for a,tag in m['tags'].items()}
            assert abs(scores['answer']['token_log_probabilities'][0]-scores['search']['token_log_probabilities'][0])<1e-3
            result=dict(label=checkpoint['label'],qid=c['qid'],condition=c['condition'],split=args.split,scores=scores,
                log_odds=scores['answer']['log_probability']-scores['search']['log_probability'],seconds=time.time()-tic,
                device='cpu',dtype='float32',cuda_initialized=False)
            append(path,result);done.add((checkpoint['label'],c['qid'],c['condition']))
            save(OUT/'status.json',dict(state='running',stage='bias_'+args.split,completed=len(done),total=total,cuda_initialized=False))
            print('BIAS_SCORE',args.split,len(done),total,checkpoint['label'],c['qid'],c['condition'],flush=True)
        del model;gc.collect()
    assert len(rows(path))==total
    print('BIAS_SCORING_COMPLETE',args.split,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'failure.json',dict(stage='score_bias',error=repr(e)));raise
