# OPD50-step GPU2/3 pilot with resource wait
Spec hash b4c8fc4e. No source/environment install changes. Modelsource remains source_v4; perrankmicro4, global128rollouts; OPDcoef1/GRPOcoef0 for botharms, teacherCE0.1/512tokens onlycandidate.

Firstwitness_opd_only failed0updates due foreignholder2274996 starting10seconds earlier on bothGPUs. Preserve ENV_WITNESS_RESOURCE_CONFLICT.md and failedrundir. NoENV_WITNESS_PASS exists. Only retriever110019 is allowed beforelaunch; waitforotherGPU2/3jobs toleave, with no arbitrary freememorythreshold. Do notkillforeignprocesses.

After HANDOFF_READY_WAIT50, fresh witnessagent executes exact command once in tmux pane%5:
```bash
/data/home/wencanning/miniconda3/envs/searchr1/bin/python /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD/research_runs/opd_pilot50_gpu23_20260920/launch/wait_and_start.py
```
This is CPUwaiting, not completedGPUvalidation. The controller runsstart.sh once whenforeignjobs disappear. Newrunname witness_opd_only_ready preservesoldattempt. Actual3stepwitness must complete, changecheckpoint, retainfinite loss/grad/LR1e-6/coef1,0/identity0/groupdiversity, alignedtrace, frozenhashes/twoworkerallocations/retrieveralive. verify_witness.py enforceschecks andwritesreleaseonlyafterpass, then50-updatequeue begins. Noautomaticretryonfailure.

Agent independentlyreviewsthe changedwrapper/verification/start/manifest and executesdocumentedinvocation, confirmsCPUwaitingstatus, preservedretriever/GPU7 and correct50stepconfig, andwrites ENV_WAIT_SETUP.md. Runtimeenvironment remainsPENDING whileGPUsblocked; do not claim actualwitness passed or manuallywriteENV_WITNESS_PASS. IfGPUsreleasebeforeyourreviewfinishes, reportactualstateandchecks. CPUguard auditscriticalmetrics continuously and FIRST_FIVE_UPDATES.json atfirst5formalupdates withtrace1/5audits. Futuremanualcheckcanverifytheseartifacts.

Formalqueue: ordinaryOPD vs sameOPD+approvedteacher-recoveryCE, paired3seeds,50actualupdates(total_training_steps51), saves25/50, same128monitor and1024expandedpanel. CPU_CHECKpassed. Primarycandidate-minusOPD comparison,extraauxiliarytokensdisclosed. Oldjoint201queuestoppedbyuserat74loggedupdates, retainstep67checkpointandstep50validation; no fabricatedstep50weights.
