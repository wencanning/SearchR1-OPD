"""Report generated-answer outcomes, never score unexecuted searches as task failure."""
import argparse
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/er_generation_cpu_20260916'


def main():
    p=argparse.ArgumentParser();p.add_argument('panel',choices=['pilot','full']);args=p.parse_args()
    out=OUT/args.panel;protocol=json.loads((OUT/'protocol.frozen.json').read_text())
    rows=[json.loads(l) for l in (out/'generations.jsonl').read_text().splitlines()]
    expected={(v,q,c,m) for v in protocol['variants'] for q in protocol[args.panel+'_qids'] for c,m in protocol['modes']}
    by={(r['variant'],r['qid'],r['condition'],r['mode']):r for r in rows}
    assert len(by)==len(rows) and set(by)==expected,'Incomplete or duplicate generations'
    assert all(not r['cuda_initialized'] for r in rows)
    summary=[];comparisons=[]
    for variant in protocol['variants']:
        for condition,mode in protocol['modes']:
            rs=[r for r in rows if r['variant']==variant and r['condition']==condition and r['mode']==mode]
            answers=[r for r in rs if r['action']=='answer']
            summary.append(dict(variant=variant,condition=condition,mode=mode,n=len(rs),
                correct_answers=sum(r['answer_em'] for r in rs),answer_actions=len(answers),
                search_actions=sum(r['action']=='search' for r in rs),invalid=sum(r['action']=='invalid' for r in rs),
                conditional_answer_em=sum(r['answer_em'] for r in answers)/len(answers) if answers else None))
            if variant=='opd':continue
            paired=[(by['opd',r['qid'],condition,mode],r) for r in rs]
            comparisons.append(dict(variant=variant,condition=condition,mode=mode,
                wrong_to_correct=sum(not a['answer_em'] and b['answer_em'] for a,b in paired),
                correct_to_wrong=sum(a['answer_em'] and not b['answer_em'] for a,b in paired),
                changed_action=sum(a['action']!=b['action'] for a,b in paired),
                changed_token_sequence=sum(a['token_ids']!=b['token_ids'] for a,b in paired)))
    report=['# 从正文概率到实际生成：单步CPU检查','',
        f"{args.panel}面板，{len(protocol[args.panel+'_qids'])}道题，{len(rows)}条生成；原8道正文探针和更新题均排除。",
        '这是受控检索证据后的生成，不是完整自然搜索任务：搜索分支没有执行。自由动作的正确答案数是立即正确回答数，不能视为最终任务EM。',
        '问题的旧动作概率曾被查看，此次按输入hash冻结选题；仍为探索性，不是新的确认测试。', '',
        '|方法|证据|生成方式|题数|正确答案数|回答动作|搜索动作|无效/未完成|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:report.append(f"|{r['variant']}|{r['condition']}|{r['mode']}|{r['n']}|{r['correct_answers']}|{r['answer_actions']}|{r['search_actions']}|{r['invalid']}|")
    report+=['','|方法vs OPD|证据|生成方式|错→对|对→错|动作变化|token序列变化|','|---|---|---|---:|---:|---:|---:|']
    for r in comparisons:report.append(f"|{r['variant']}|{r['condition']}|{r['mode']}|{r['wrong_to_correct']}|{r['correct_to_wrong']}|{r['changed_action']}|{r['changed_token_sequence']}|")
    forced=[r for r in comparisons if r['mode']=='forced_answer']
    if all(r['wrong_to_correct']==r['correct_to_wrong']==0 for r in forced):
        report+=['','本面板未观察到强制直接回答正确性的变化。停止用本批次单步logp增益宣称答案变好；结果不能排除更充分训练的作用。']
    report+=['','所有变体同一OPD step50起点、同32条冻结短续写、fresh AdamW、GRPO优势0；用历史梯度范数检查更新复现。',
        '小样本、单次更新和确定性解码不支持长期训练收益结论；后续扩大实验由用户手动启动，不能按pilot结果重新选题。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,paired=comparisons),indent=2)+'\n')
    print('GENERATION_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
