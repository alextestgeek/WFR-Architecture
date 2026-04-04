#requires -Version 5.1
<#
  Загрузка минимального дерева репозитория на удалённый GPU (PuTTY pscp/plink).
  Структура как в git: корень репо + пакет wfr/ (ядро) + experiments/...; иначе ``import wfr.core`` не сработает.

  Переменные окружения (все обязательны; секреты — только локально, см. ваш внутренний RULES.md если есть):
    WFR_SSH_PASSWORD, WFR_SSH_HOSTKEY, WFR_SSH_HOST, WFR_SSH_USER
    WFR_SSH_REMOTE_PARENT — абсолютный путь к корню копии репо на сервере (например /home/you/project)

  Пример:
    $env:WFR_SSH_PASSWORD = "..."
    $env:WFR_SSH_HOSTKEY = "ssh-ed25519 255 SHA256:..."
    $env:WFR_SSH_HOST = "<host>"
    $env:WFR_SSH_USER = "<user>"
    $env:WFR_SSH_REMOTE_PARENT = "/home/<user>/<repo-dir>"
    .\experiments\08-wikitext-rfp\remote_sync_wikitext.ps1 -Direction upload

  На сервере после upload (войти в тот же каталог, что в WFR_SSH_REMOTE_PARENT):
    cd <remote-repo-root> && .venv/bin/python -u experiments/08-wikitext-rfp/run_wikitext_train.py --quick
  Exp 09 parity: bash experiments/09-lm-parity/remote_gpu_parity.sh (см. experiments/09-lm-parity/README.md)
  Быстрая проверка CUDA + parity (чеклист docs/07 шаг 9): из корня копии на сервере — bash _remote_gpu_check.sh
  Для nohup: ``.venv/bin/python -u ...`` или PYTHONUNBUFFERED=1; см. docstring run_wikitext_train.py.
  Другой корень на сервере: -RemoteParent "/home/USER/Desktop/WFR-Sync"
  (предварительно: venv + pip install datasets torch; python data/hf/download_wikitext2.py)
#>

param(
    [ValidateSet("upload", "download")]
    [string]$Direction = "upload",
    [string]$RootLocal = "",
    [string]$RemoteParent = ""
)

$ErrorActionPreference = "Stop"
if (-not $RootLocal) { $RootLocal = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
if (-not $RemoteParent) { $RemoteParent = $env:WFR_SSH_REMOTE_PARENT }
if (-not $RemoteParent) {
    Write-Error "Set -RemoteParent or env WFR_SSH_REMOTE_PARENT (absolute path to repo root on the server)."
}

$pw = $env:WFR_SSH_PASSWORD
if (-not $pw) { Write-Error "Set WFR_SSH_PASSWORD" }

$hostName = $env:WFR_SSH_HOST
if (-not $hostName) { Write-Error "Set WFR_SSH_HOST" }
$user = $env:WFR_SSH_USER
if (-not $user) { Write-Error "Set WFR_SSH_USER" }
$hostKey = $env:WFR_SSH_HOSTKEY
if (-not $hostKey) { Write-Error "Set WFR_SSH_HOSTKEY" }

$pscp = "C:\Program Files\PuTTY\pscp.exe"
$plink = "C:\Program Files\PuTTY\plink.exe"
if (-not (Test-Path $pscp)) { Write-Error "PuTTY not found: $pscp" }

$remoteBase = "${user}@${hostName}:${RemoteParent}"

function Upload([string]$Local, [string]$Remote) {
    & $pscp -pw $pw -hostkey $hostKey $Local $Remote
    if ($LASTEXITCODE -ne 0) { throw "pscp failed: $Local" }
}

if ($Direction -eq "upload") {
    $mk = "mkdir -p '$RemoteParent/wfr' '$RemoteParent/experiments/00-smoke-test' '$RemoteParent/experiments/06-rfp-v0' '$RemoteParent/experiments/08-wikitext-rfp' '$RemoteParent/experiments/09-lm-parity' '$RemoteParent/experiments/09-lm-parity/outputs/remote_a100/runs' '$RemoteParent/data/hf' '$RemoteParent/outputs'"
    & $plink -ssh -batch -pw $pw -hostkey $hostKey "${user}@${hostName}" $mk
    if ($LASTEXITCODE -ne 0) { throw "plink mkdir failed" }

    Upload (Join-Path $RootLocal "wfr_lm.py") "$remoteBase/wfr_lm.py"
    Upload (Join-Path $RootLocal "wfr_rfp.py") "$remoteBase/wfr_rfp.py"
    Upload (Join-Path $RootLocal "wfr\__init__.py") "$remoteBase/wfr/__init__.py"
    Upload (Join-Path $RootLocal "wfr\core.py") "$remoteBase/wfr/core.py"
    Upload (Join-Path $RootLocal "wfr\losses.py") "$remoteBase/wfr/losses.py"
    Upload (Join-Path $RootLocal "experiments\00-smoke-test\phase0_best_config.py") "$remoteBase/experiments/00-smoke-test/phase0_best_config.py"
    Upload (Join-Path $RootLocal "experiments\06-rfp-v0\run_rfp_training.py") "$remoteBase/experiments/06-rfp-v0/run_rfp_training.py"
    Upload (Join-Path $PSScriptRoot "wikitext_loader.py") "$remoteBase/experiments/08-wikitext-rfp/wikitext_loader.py"
    Upload (Join-Path $PSScriptRoot "run_wikitext_train.py") "$remoteBase/experiments/08-wikitext-rfp/run_wikitext_train.py"
    Upload (Join-Path $PSScriptRoot "remote_gpu_wikipush.sh") "$remoteBase/experiments/08-wikitext-rfp/remote_gpu_wikipush.sh"
    Upload (Join-Path $PSScriptRoot "wait_wikitext_python.sh") "$remoteBase/experiments/08-wikitext-rfp/wait_wikitext_python.sh"
    Upload (Join-Path $PSScriptRoot "test_input_schemes.py") "$remoteBase/experiments/08-wikitext-rfp/test_input_schemes.py"
    Upload (Join-Path $PSScriptRoot "run_theory_phase2.py") "$remoteBase/experiments/08-wikitext-rfp/run_theory_phase2.py"
    Upload (Join-Path $PSScriptRoot "test_wikitext_ab.py") "$remoteBase/experiments/08-wikitext-rfp/test_wikitext_ab.py"
    Upload (Join-Path $PSScriptRoot "test_wikitext_smoke.py") "$remoteBase/experiments/08-wikitext-rfp/test_wikitext_smoke.py"
    Upload (Join-Path $PSScriptRoot "_remote_gpu_check.sh") "$remoteBase/_remote_gpu_check.sh"
    Upload (Join-Path $PSScriptRoot "_remote_pipeline_p1_p2_p3.sh") "$remoteBase/_remote_pipeline_p1_p2_p3.sh"
    Upload (Join-Path $RootLocal "data\hf\download_wikitext2.py") "$remoteBase/data/hf/download_wikitext2.py"
    $exp09 = Join-Path $RootLocal "experiments\09-lm-parity"
    Upload (Join-Path $exp09 "run_parity_pair.py") "$remoteBase/experiments/09-lm-parity/run_parity_pair.py"
    Upload (Join-Path $exp09 "run_longcontext_pair.py") "$remoteBase/experiments/09-lm-parity/run_longcontext_pair.py"
    Upload (Join-Path $exp09 "run_transformer_char_baseline.py") "$remoteBase/experiments/09-lm-parity/run_transformer_char_baseline.py"
    Upload (Join-Path $exp09 "parity_capacity.py") "$remoteBase/experiments/09-lm-parity/parity_capacity.py"
    Upload (Join-Path $exp09 "remote_gpu_parity.sh") "$remoteBase/experiments/09-lm-parity/remote_gpu_parity.sh"
    Upload (Join-Path $exp09 "remote_gpu_b3_confirm.sh") "$remoteBase/experiments/09-lm-parity/remote_gpu_b3_confirm.sh"
    Upload (Join-Path $exp09 "remote_gpu_b1_mlp_matrix.sh") "$remoteBase/experiments/09-lm-parity/remote_gpu_b1_mlp_matrix.sh"
    Upload (Join-Path $exp09 "remote_gpu_b3_combo_d32.sh") "$remoteBase/experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh"
    Upload (Join-Path $exp09 "remote_gpu_longcontext_d32.sh") "$remoteBase/experiments/09-lm-parity/remote_gpu_longcontext_d32.sh"
    Upload (Join-Path $exp09 "summarize_parity_json.py") "$remoteBase/experiments/09-lm-parity/summarize_parity_json.py"
    Upload (Join-Path $exp09 "verify_theory_calibration.py") "$remoteBase/experiments/09-lm-parity/verify_theory_calibration.py"
    Upload (Join-Path $exp09 "verify_longcontext_artifacts.py") "$remoteBase/experiments/09-lm-parity/verify_longcontext_artifacts.py"
    Write-Host "OK: tree under $RemoteParent (activate venv on the server before training)."
}
else {
    $outLocal = Join-Path $PSScriptRoot "outputs"
    New-Item -ItemType Directory -Force -Path $outLocal | Out-Null
    & $pscp -pw $pw -hostkey $hostKey "$remoteBase/experiments/08-wikitext-rfp/outputs/*" "$outLocal\"
    if ($LASTEXITCODE -ne 0) { Write-Warning "Download exit code $LASTEXITCODE (folder may be empty)" }
    $out09 = Join-Path $RootLocal "experiments\09-lm-parity\outputs\remote_a100"
    New-Item -ItemType Directory -Force -Path $out09 | Out-Null
    & $pscp -pw $pw -hostkey $hostKey "$remoteBase/experiments/09-lm-parity/outputs/*" "$out09\"
    if ($LASTEXITCODE -ne 0) { Write-Warning "Exp09 download exit code $LASTEXITCODE" }
    Write-Host "Downloaded Exp08 -> $outLocal ; Exp09 -> $out09"
}
