# Rewrite the GitHub repository content in ONE commit: delete the upstream leftovers
# (config/, examples/, best_model_reg95.pth) and upload the cleaned project.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/update_github_repo.ps1 `
#       -Token "github_pat_xxx" -Owner "Xhw07" -Repo "MobileVitv3-DSR"
#
# The token needs "Contents: Read and write" on that repository (fine-grained token),
# or classic token with "repo" scope. Revoke it when you are done.

param(
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$Owner = "Xhw07",
    [string]$Repo = "MobileVitv3-DSR",
    [string]$Branch = "main",
    [string]$SourceDir = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$api = "https://api.github.com"
$headers = @{
    Authorization          = "Bearer $Token"
    Accept                 = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent"           = "MobileVitv3-DSR-repo-updater"
}

function Invoke-GitHub {
    param([string]$Method, [string]$Path, $Body)
    $uri = "$api$Path"
    if ($Body) {
        $json = $Body | ConvertTo-Json -Depth 12 -Compress
        return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) `
            -ContentType "application/json; charset=utf-8"
    }
    return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers
}

$repoPath = "/repos/$Owner/$Repo"

Write-Host "reading current $Branch tree ..." -ForegroundColor Cyan
$ref = Invoke-GitHub GET "$repoPath/git/ref/heads/$Branch"
$commitSha = $ref.object.sha
$commit = Invoke-GitHub GET "$repoPath/git/commits/$commitSha"
$baseTree = $commit.tree.sha
$existing = (Invoke-GitHub GET "$repoPath/git/trees/$baseTree`?recursive=1").tree

# paths to remove: the upstream MobileViT dump plus stale artifacts
$dropPrefixes = @("config/", "examples/")
$dropExact = @("best_model_reg95.pth")
$deletes = @()
foreach ($item in $existing) {
    if ($item.type -ne "blob") { continue }
    $p = $item.path
    $kill = $false
    foreach ($pre in $dropPrefixes) { if ($p.StartsWith($pre)) { $kill = $true } }
    if ($dropExact -contains $p) { $kill = $true }
    if ($kill) { $deletes += $p }
}
Write-Host ("deleting {0} files (config/, examples/, stale weights)" -f $deletes.Count) -ForegroundColor Yellow

# files to upload: everything in SourceDir except ignored artifacts
$ignoreDirs = @("runs", "__pycache__", ".git", "data", ".idea", ".vscode")
$localFiles = Get-ChildItem -LiteralPath $SourceDir -Recurse -File | Where-Object {
    $rel = $_.FullName.Substring($SourceDir.Length + 1).Replace("\", "/")
    if ($rel -match "__pycache__" -or $rel -like "*.pyc") { return $false }
    if ($rel -like "*.pth" -or $rel -like "*.pt") { return $false }
    $top = ($rel -split "/")[0]
    if ($ignoreDirs -contains $top) { return $false }
    return $true
}

$binaryExt = @(".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".pdf", ".bmp")

$entries = @()
foreach ($d in $deletes) {
    $entries += @{ path = $d; mode = "100644"; type = "blob"; sha = $null }
}

foreach ($f in $localFiles) {
    $rel = $f.FullName.Substring($SourceDir.Length + 1).Replace("\", "/")
    $ext = $f.Extension.ToLower()
    if ($binaryExt -contains $ext) {
        $content = [Convert]::ToBase64String([IO.File]::ReadAllBytes($f.FullName))
        $blob = Invoke-GitHub POST "$repoPath/git/blobs" @{ content = $content; encoding = "base64" }
    } else {
        $text = [IO.File]::ReadAllText($f.FullName, [Text.Encoding]::UTF8)
        $blob = Invoke-GitHub POST "$repoPath/git/blobs" @{ content = $text; encoding = "utf-8" }
    }
    $mode = "100644"
    if ($rel -like "scripts/*.ps1") { }  # regular files are fine
    $entries += @{ path = $rel; mode = $mode; type = "blob"; sha = $blob.sha }
    Write-Host ("  + {0}" -f $rel)
}

if ($DryRun) {
    Write-Host "dry run: $($entries.Count) tree entries prepared, nothing pushed" -ForegroundColor Yellow
    return
}

Write-Host "creating tree ..." -ForegroundColor Cyan
$tree = Invoke-GitHub POST "$repoPath/git/trees" @{ base_tree = $baseTree; tree = $entries }

$message = @"
chore: rebuild repo as a runnable MobileViTv3-DSR project

- remove the upstream MobileViT dump (config/, examples/range_augment)
- remove stale weights (best_model_reg95.pth); publish weights via Release instead
- add models/ (real SRA + differential attention), utils/, argparse train/test
- deterministic training/eval (fixed seed, no random ops in val/test)
- rewrite README with reproducible numbers, figures, license and citation
"@

Write-Host "creating commit ..." -ForegroundColor Cyan
$newCommit = Invoke-GitHub POST "$repoPath/git/commits" @{
    message = $message
    tree    = $tree.sha
    parents = @($commitSha)
}

Write-Host "updating ref $Branch ..." -ForegroundColor Cyan
$null = Invoke-GitHub PATCH "$repoPath/git/refs/heads/$Branch" @{ sha = $newCommit.sha; force = $false }

Write-Host "updating repo description / topics ..." -ForegroundColor Cyan
$null = Invoke-GitHub PATCH $repoPath @{
    description = "Lightweight traffic-sign classification: MobileViTv3 with differential attention (DA) and spatial-reduction attention (SRA)."
    topics      = @("traffic-sign-recognition", "mobilevit", "differential-attention", "lightweight-models", "pytorch", "gtsrb")
}

Write-Host ""
Write-Host ("done. new commit {0}" -f $newCommit.sha) -ForegroundColor Green
Write-Host ("open: https://github.com/$Owner/$Repo/commit/{0}" -f $newCommit.sha)
