# Post-repair experiment queue, unattended and resume-safe.
#
#   1. stage-rule check   ~15 min  CPU    (already done -> skips finished rows)
#   2. full-system pass   ~10 min  GPU    validates pipeline 2 BEFORE the long run
#   3. main ablation      ~30 h    GPU    4 arms x 3 seeds, headline numbers
#   4. PUCT P-battery     ~2 h     GPU    search vs no search, production model
#   5. rescue R/RX        ~1 h     GPU    weak proposer + LLM-failure specs
#   6. sizing baselines   ~35 min  CPU    C9 vs grid / TPE / old SAC
#
# Step 2 is deliberately before the 30-hour run: run_full_raptor.py was
# rewritten today (real spec into the search, 5-structure pool, spec-ranked
# retrieval) and has not executed since. Ten minutes here beats discovering a
# broken act-on path on Wednesday.
#
# Steps need the GPU, so they run one at a time. Every step resumes on its
# own, so re-running this script after any interruption continues rather
# than restarting.
#
# Logs + live status: artifacts\run_all\<timestamp>\
#
# Run:  powershell -ExecutionPolicy Bypass -File run_all.ps1

$ErrorActionPreference = 'Continue'
$root = 'C:\Users\kobeo\OneDrive\Desktop\raptor1\Agentic_Raptor'
$py = 'C:\Users\kobeo\AppData\Local\Python\pythoncore-3.14-64\python.exe'
Set-Location $root
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logdir = Join-Path $root "artifacts\run_all\$stamp"
New-Item -ItemType Directory -Force $logdir | Out-Null
$status = Join-Path $logdir 'STATUS.txt'

function Write-Status($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $status -Value $line -Encoding utf8
}

function Invoke-Step($name, $scriptArgs) {
    $out = Join-Path $logdir "$name.log"
    $err = Join-Path $logdir "$name.err.log"
    Write-Status "START  $name"
    $t0 = Get-Date
    $p = Start-Process -FilePath $py -ArgumentList $scriptArgs `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($p.ExitCode -eq 0) {
        Write-Status "DONE   $name  ($mins min)"
    } else {
        Write-Status "FAILED $name  (exit $($p.ExitCode), $mins min) - see $name.err.log"
    }
    return $p.ExitCode
}

Write-Status "queue started; logs in $logdir"

# 1. Is the corpus stage rule supported by measurement? Already answered
#    (2-stage closer on 6/8); finished rows are skipped on re-run.
Invoke-Step 'step1_stage_rule' @('check_stage_rule.py', '--specs', '8') | Out-Null

# 2. One full-system pass. Cheap insurance on today's pipeline-2 rewrite.
#    It is the only step with no internal resume, so skip it once it has
#    produced a trace -- otherwise every restart pays 10 minutes again.
#    Skip only when the trace is NEWER than the script that produced it --
#    a trace from before today's rewrite proves nothing about today's code.
$trace = Join-Path $root 'artifacts\full_raptor_run\TRACE.json'
$src = Join-Path $root 'run_full_raptor.py'
$fresh = (Test-Path $trace) -and
         ((Get-Item $trace).LastWriteTime -gt (Get-Item $src).LastWriteTime)
if ($fresh) {
    Write-Status "SKIP   step2_full_raptor (trace newer than script)"
} else {
    Invoke-Step 'step2_full_raptor' @('run_full_raptor.py') | Out-Null
}

# 3. Headline experiment. Longer than before: the search now ranks 5
#    structures and sends 2 per spec to SAC, so ~24 structures are sized per
#    generation instead of ~3.
#    TWO seeds, all FOUR arms. Dropping a seed costs statistical confidence;
#    dropping an arm would delete part of the ablation itself, so seeds are
#    the right thing to trade. Add 47 back for the third seed (~+10 h).
Invoke-Step 'step3_main_ablation' @('run_ablation.py', '--seeds', '11,23') | Out-Null

# 4. Search vs no search on the production model, clean 29 specs, all arms.
Invoke-Step 'step4_puct_battery' @('run_puct_ablation.py', '--tasks', '29') | Out-Null

# 5. Weak proposer, including the specs the LLM cannot answer at all.
Invoke-Step 'step5_rescue' @('run_puct_rescue.py') | Out-Null

# 6. SIZING BASELINES (C-battery): does RL sizing beat classical optimisation
#    on an equal SPICE budget? Grid (C2) and TPE (C3) are the honest rivals;
#    C8 is the pre-repair engine; C9/C9s are the hybrid sizer in isolation --
#    no LLM, no search, no RAG. The stored results predate both the smooth
#    deficit reward and the widened SAC state, so they cannot speak to the
#    current sizer. Same 12 frozen tasks and budget as that run, for
#    continuity. ~35 min at 4 workers.
$sb = Join-Path $root 'artifacts\publication\sizing_baselines.json'
if (Test-Path $sb) {
    $arc = Join-Path $root ("artifacts\publication\sizing_baselines.pre_" +
                            "reward_v2_{0}.json" -f $stamp)
    Move-Item $sb $arc -Force
    Write-Status "archived stale sizing baselines -> $(Split-Path $arc -Leaf)"
}
Invoke-Step 'step6_sizing_baselines' @(
    '-m', 'agentic_raptor.publication.sizing_baselines',
    'T006,T011,T021,T030,T042,T076,T085,T086,T097,T100,T103,T104',
    '--methods', 'C0,C2,C3,C4,C6,C7,C8,C9,C9s',
    '--workers', '4') | Out-Null

Write-Status 'queue finished'
foreach ($s in @(
    'artifacts\publication\stage_rule_check\SUMMARY.json',
    'artifacts\publication\puct_ablation\SUMMARY.json',
    'artifacts\publication\puct_rescue\SUMMARY.json')) {
    if (Test-Path $s) { Write-Status "summary: $s" }
}
Write-Status 'aggregate campaigns: python -m agentic_raptor.publication.aggregate'
