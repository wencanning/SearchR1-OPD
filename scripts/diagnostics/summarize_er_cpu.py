"""Summarize ONLY the full-forward v2 probe, never invalid cached pilot scores."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/offline_er_cpu_v3'
TARGETS=['observed','er_0.5','er_1','er_1.5','temperature_0.7','temperature_0.5']


def main():
    rows=[json.loads(l) for l in (OUT/'scores.jsonl').read_text().splitlines()]
    def margin(r,t,metric='sum'):
        return r['scores']['gold'][t][metric]-r['scores']['competitor'][t][metric]
    groups={'audited_failure':[r for r in rows if r['group']=='audited_failure'],
            'clean_correct_control':[r for r in rows if r['group']=='EM_correct_control' and not r['key'].endswith(':49679')],
            'ambiguous_control':[r for r in rows if r['key'].endswith(':49679')]}
    summary={}
    lines=['# CPU-only ER probe: eager full-forward v3',
      '', 'Positive margin = higher teacher-forced likelihood for gold than competitor.',
      'These are selected fixed-prefix cases, NOT generation accuracy or a training trial.',
      'All forwards CPU FP32 eager, use_cache=False. Prefix contains original student reasoning.',
      'First cached pilot was invalidated; no figures below use it.', '']
    for metric in ['sum','mean']:
        lines+=['## '+metric+' log-probability margins','',
                '| key | gold / competitor | '+' | '.join(TARGETS)+' |',
                '|---|---|'+'---:|'*len(TARGETS)]
        for r in rows:
            lines.append('|'+r['key']+'|'+r['candidates']['gold']+' / '+r['candidates']['competitor']+'|'+
                         '|'.join(f'{margin(r,t,metric):.3f}' for t in TARGETS)+'|')
    for name,rs in groups.items():
        summary[name]={'n':len(rs)}
        for t in TARGETS:
            summary[name][t]={
              'gold_preferred_sum':sum(margin(r,t)>0 for r in rs),
              'gold_preferred_mean':sum(margin(r,t,'mean')>0 for r in rs),
              'margin_improved_sum':sum(margin(r,t)>margin(r,'observed')+1e-5 for r in rs),
              'margin_improved_mean':sum(margin(r,t,'mean')>margin(r,'observed','mean')+1e-5 for r in rs),
              'wrong_to_gold_sum':sum(margin(r,'observed')<=0<margin(r,t) for r in rs),
              'gold_to_wrong_sum':sum(margin(r,t)<=0<margin(r,'observed') for r in rs)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines+=['','## Limitations','',
      '- Tiny selected sample: no population rates, significance or general superiority claims.',
      '- Armenian-American vs American is ambiguous and excluded from clean controls.',
      '- Malformed information tags excluded without replacement.',
      '- Candidate sums include closing answer delimiter; mean scores are a length sensitivity check.',
      '- Temperature controls are NOT entropy matched; do not claim complete separation from sharpening.',
      '- FP32 CPU inference differs numerically from FP16 training; no student update evaluated.',
      '- No fabricated distractor retrieval / causal treatment of student reasoning was tested.',
      '- Baseline favors gold does not guarantee the student will learn it.',
      '- Full manifest and candidates: manifest.json; flushed raw scores: scores.jsonl.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
