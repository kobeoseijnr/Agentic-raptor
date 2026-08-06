# Case C gates, in one command.
#
#   Gate A  family realizability      ~10 min  CPU/ngspice
#           Are >= 5 structure families actually buildable and measurable?
#           If not, training a proposer to emit 5 is pointless -- Step 2 of
#           the Case C plan makes this a hard blocker.
#
#   Gate B  SFT-only proposer diversity  ~70 min  GPU
#           The earlier run used --arm L8, which resolves to gen2_DPO. The
#           spec designates the accepted SFT-only proposer as the primary
#           subject, with DPO as a diagnostic control only. L6 = gen2_sft.
#           This decides Case B (drop proposer-side DPO, no retraining)
#           versus Case C (rebuild corpus + retrain).
#
# Gate A is CPU-only so it starts immediately even if a GPU job is active.
# Gate B waits for any running check_diversity to finish first, so the two
# never contend for the GPU.
#
# Run:  powershell -ExecutionPolicy Bypass -File run_case_c_gates.ps1

$ErrorActionPreference = 'Continue'
$root = 'C:\Users\kobeo\OneDrive\Desktop\raptor1\Agentic_Raptor'
$py = 'C:\Users\kobeo\AppData\Local\Python\pythoncore-3.14-64\python.exe'
Set-Location $root
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = $root

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logdir = Join-Path $root "artifacts\publication_v2\proposer_repair\gates_$stamp"
New-Item -ItemType Directory -Force $logdir | Out-Null
$status = Join-Path $logdir 'STATUS.txt'

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $status -Value $line -Encoding utf8
}

function Step($name, $argList) {
    $out = Join-Path $logdir "$name.log"
    $err = Join-Path $logdir "$name.err.log"
    Say "START  $name"
    $t0 = Get-Date
    $p = Start-Process -FilePath $py -ArgumentList $argList `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($p.ExitCode -eq 0) { Say "DONE   $name ($mins min)" }
    else { Say "FAILED $name (exit $($p.ExitCode), $mins min) - see $name.err.log" }
    return $p.ExitCode
}

Say "Case C gates; logs in $logdir"

# ---- Gate A: family realizability (CPU) --------------------------------
Step 'gateA_family_reference' @('run_family_reference.py', '--budget', '24') | Out-Null

$fam = Join-Path $root 'artifacts\publication_v2\family_reference\SUMMARY.json'
$realizable = 0
if (Test-Path $fam) {
    $j = Get-Content $fam -Raw | ConvertFrom-Json
    $realizable = @($j.resolved).Count
    Say "realizable families: $realizable  -> $($j.resolved -join ', ')"
    if (@($j.unresolved).Count -gt 0) {
        Say "UNRESOLVED: $($j.unresolved -join ', ')"
    }
    if ($realizable -lt 5) {
        Say "PROPOSER TARGET SPACE INSUFFICIENT: True"
        Say "Fewer than 5 realizable families. Do NOT retrain the proposer to"
        Say "emit structures the mapper/testbench cannot realise -- expand the"
        Say "topology grammar or realization library first."
    } else {
        Say "PROPOSER TARGET SPACE INSUFFICIENT: False"
    }
} else {
    Say "gate A produced no summary"
}

# ---- Gate B: SFT-only diversity (GPU) ----------------------------------
# never contend with an in-flight GPU job
while (@(Get-CimInstance Win32_Process |
         Where-Object { $_.CommandLine -match 'check_diversity' }).Count -gt 0) {
    Say 'waiting: another check_diversity run holds the GPU'
    Start-Sleep -Seconds 120
}
Step 'gateB_diversity_sft_only' @(
    'check_diversity.py', '--specs', '6', '--target', '5', '--arm', 'L6') | Out-Null

$div = Join-Path $root 'artifacts\publication_v2\diversity\SUMMARY.json'
if (Test-Path $div) {
    Copy-Item $div (Join-Path $logdir 'diversity_L6_SUMMARY.json') -Force
    $d = Get-Content $div -Raw | ConvertFrom-Json
    Say "arm=$($d.arm) reaching_target=$($d.specs_reaching_target)/$($d.specs) ladder_mean=$($d.ladder_mean_distinct)"
    if ($d.specs_reaching_target -eq $d.specs) {
        Say 'PROPOSER DIVERSITY PASS: True  -> CASE B (SFT is fine; proposer-side DPO collapsed diversity)'
    } else {
        Say 'PROPOSER DIVERSITY PASS: False -> CASE C (retrain SFT on corpus_diverse.json)'
    }
}
Say 'gates finished'
