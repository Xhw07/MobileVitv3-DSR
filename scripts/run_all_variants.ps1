# Run the full ablation study (baseline / sra / da / dsr) and evaluate each model.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/run_all_variants.ps1 -DataRoot "D:/data/GTSRB4" -Epochs 50

param(
    [string]$DataRoot = "data/GTSRB4",
    [int]$Epochs = 50,
    [int]$ImgSize = 96,
    [int]$BatchSize = 32,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

foreach ($variant in @("baseline", "sra", "da", "dsr")) {
    Write-Host "=== training $variant ===" -ForegroundColor Cyan
    & $Python train.py --data-root $DataRoot --variant $variant `
        --epochs $Epochs --img-size $ImgSize --batch-size $BatchSize --out-dir "runs/$variant"
    if ($LASTEXITCODE -ne 0) { throw "training failed for $variant" }

    Write-Host "=== evaluating $variant ===" -ForegroundColor Cyan
    & $Python test.py --data-root $DataRoot --checkpoint "runs/$variant/best.pth" --split test
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed for $variant" }
}

Write-Host "done. per-variant metrics are in runs/<variant>/metrics_test.json" -ForegroundColor Green
