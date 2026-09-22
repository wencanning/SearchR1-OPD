"""CPU check for reusing the measured native unupdated-model baseline."""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from probe import ROOT,STUDENT,save
torch.set_num_threads(4)
OUT=ROOT/'reports/recovery_20260917_native'


def main():
    zero=OUT/'update_base/zero_step_checkpoint'
    configs=[json.loads((p/'config.json').read_text()) for p in [STUDENT,zero]]
    differences={k:[configs[0].get(k),configs[1].get(k)] for k in configs[0].keys()|configs[1].keys() if configs[0].get(k)!=configs[1].get(k)}
    assert set(differences)<= {'_name_or_path','torch_dtype','use_cache','transformers_version'},differences
    assert configs[0]['transformers_version']==configs[1]['transformers_version']
    # The decoder explicitly loads BF16 and sets use_cache=True; names are metadata.
    assert json.loads((STUDENT/'generation_config.json').read_text())==json.loads((zero/'generation_config.json').read_text())
    a=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True);b=AutoTokenizer.from_pretrained(zero,local_files_only=True)
    assert a.get_vocab()==b.get_vocab() and a.chat_template==b.chat_template and a.special_tokens_map==b.special_tokens_map
    def index(path):
        result={}
        for file in path.glob('*.safetensors'):
            with safe_open(file,framework='pt',device='cpu') as f:
                for key in f.keys():assert key not in result;result[key]=file
        return result
    ix,jx=index(STUDENT),index(zero);keys=set(ix)|set(jx);assert keys
    if configs[0].get('tie_word_embeddings'):
        for mapping in [ix,jx]:
            if 'lm_head.weight' not in mapping:assert 'model.embed_tokens.weight' in mapping
    count=0
    for key in sorted(keys):
        actual=[]
        for mapping in [ix,jx]:
            k=key
            if k not in mapping:
                assert k=='lm_head.weight' and configs[0].get('tie_word_embeddings');k='model.embed_tokens.weight'
            with safe_open(mapping[k],framework='pt',device='cpu') as f:actual.append(f.get_tensor(k).to(torch.bfloat16))
        assert torch.equal(actual[0],actual[1]),key;count+=actual[0].numel()
    status=json.loads((OUT/'continue_student/status.json').read_text());assert status['state']=='complete'
    result=dict(state='complete',baseline_reused=True,checkpoint=str(STUDENT),zero_checkpoint=str(zero),equal_tensor_keys=len(keys),equal_elements=count,effective_inference_dtype='bfloat16',config_differences=differences,overrides='decoder explicitly loads BF16 and use_cache=True; _name_or_path only metadata',generation_config_equal=True,tokenizer_equal=True,baseline_source_sha256=hashlib.sha256((OUT/'continue_student/continuations.jsonl').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='CPU inference-state equivalence check, NOT new zero-step generation; native original-checkpoint heldout outcomes are reused. Prior strict-parser zero-step actual regeneration was16/16 exact.')
    save(OUT/'BASELINE_REUSE_EQUIVALENCE.json',result);print(json.dumps(result))


if __name__=='__main__':main()
