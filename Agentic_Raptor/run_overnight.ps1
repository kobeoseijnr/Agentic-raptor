# Overnight orchestrator: wait for the running DPO to finish, ensure it
# completed, then run the Qwen3-VL multimodal pilot. Logs to artifacts\overnight\
Set-Location "C:\Users\kobeo\OneDrive\Desktop\raptor1\Agentic_Raptor"
$py = "C:\Users\kobeo\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$log = "artifacts\overnight"
New-Item -ItemType Directory -Force $log | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
"started $(Get-Date)" | Out-File "$log\status.txt" -Encoding utf8

# 1) Wait (up to 90 min) for the GPU to free up = user's DPO run finishing
$deadline = (Get-Date).AddMinutes(90)
while ((Get-Date) -lt $deadline) {
    $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if ($used -lt 2500) { break }
    Start-Sleep -Seconds 60
}
"gpu free $(Get-Date)" | Out-File "$log\status.txt" -Append -Encoding utf8

# 2) If the DPO leg never completed, run it here
if (-not (Test-Path "artifacts\stage3e4\dpo_qwen_result.json")) {
    "running DPO $(Get-Date)" | Out-File "$log\status.txt" -Append -Encoding utf8
    $env:AGENTIC_RAPTOR_TOPOLOGY_LLM = 'Qwen/Qwen2.5-3B-Instruct'
    & $py run_dpo_qwen.py *> "$log\dpo_qwen.log"
}
"dpo done $(Get-Date)" | Out-File "$log\status.txt" -Append -Encoding utf8

# 3) Multimodal pilot on the freed GPU
$env:AGENTIC_RAPTOR_VLM = 'Qwen/Qwen3-VL-4B-Instruct'
& $py -m agentic_raptor.llm_dpo.multimodal *> "$log\multimodal.log"
"multimodal done $(Get-Date) exit=$LASTEXITCODE" | Out-File "$log\status.txt" -Append -Encoding utf8
