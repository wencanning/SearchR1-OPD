"""CPU analysis only after all raw collection stages pass completion checks."""
import json
from pathlib import Path
import analyze_recovery


def main():
    analyze_recovery.main()
    out = analyze_recovery.OUT
    for phase in ('collected','continue_student','continue_teacher'):
        s=json.loads((out/phase/'status.json').read_text())
        assert s['state']=='complete' and s['completed']==s['total']
    # EM-selected suffixes are provisional data, not automatically cleared training targets.
    (out/'TARGET_REVIEW_REQUIRED.md').write_text(
        'Only train-question diagnostic collection completed. References were preserved from the source dataset. '
        'Before auxiliary training, review source labels/aliases and provenance under a fixed output-blind policy. '
        'No full online training or heldout efficacy evaluation has been launched. '
        'Input selection and bounded native continuation protocol are in train_inputs.frozen.json.\n')
    (out/'analysis_status.json').write_text(json.dumps(dict(state='complete',completed=1,total=1))+'\n')


if __name__=='__main__': main()
