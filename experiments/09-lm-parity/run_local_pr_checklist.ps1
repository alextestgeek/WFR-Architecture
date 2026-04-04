# Локальный чеклист перед PR / длинным GPU (docs/07-experiment-plan.md, шаги 1–6; опции 7–8).
# Запуск из любого каталога:
#   powershell -ExecutionPolicy Bypass -File experiments/09-lm-parity/run_local_pr_checklist.ps1
# Опции:
#   -SkipOptional   — не запускать шаги 7–8 (stability spot-tests, wikitext smoke)
param(
    [switch]$SkipOptional
)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Step($n, [scriptblock]$fn) {
    Write-Host ""
    Write-Host "=== Step $n ===" -ForegroundColor Cyan
    & $fn
    if ($LASTEXITCODE -ne 0) { throw "Step $n failed with exit $LASTEXITCODE" }
}

try {
    Step 1 { python -m compileall -q wfr wfr_lm.py wfr_rfp.py }
    Step 2 { python experiments/00-smoke-test/run_smoke_test.py }
    Step 3 { python -m pytest -q --tb=short }
    Step 4 { python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity }
    Step 5 { python experiments/09-lm-parity/run_transformer_char_baseline.py --quick }
    Step 6 { python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity --readout-feat-dim 16 }

    if (-not $SkipOptional) {
        Step "7 (optional)" {
            python -m pytest `
                experiments/03-long-context-stability/run_stability_test.py::test_1_determinism `
                experiments/03-long-context-stability/run_stability_test.py::test_3_cross_context_standing_wave `
                -q --tb=short
        }
        Step "8 (optional)" {
            python -m pytest experiments/08-wikitext-rfp/test_wikitext_smoke.py -q --tb=short
        }
    }

    Step "verify H1/B3" { python experiments/09-lm-parity/verify_theory_calibration.py }
    Step "verify D.1 JSON" { python experiments/09-lm-parity/verify_longcontext_artifacts.py }

    Write-Host ""
    Write-Host "All checklist steps passed." -ForegroundColor Green
    Write-Host "Next GPU stages: experiments/09-lm-parity/NEXT_PHASE_RUNBOOK.md (P1 combo, P3 longcontext)."
}
catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
