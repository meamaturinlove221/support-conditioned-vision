param(
    [string]$Repository = "https://github.com/meamaturinlove221/support-conditioned-vision.git"
)
$ErrorActionPreference = "Stop"
$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$Git = Get-Command git -ErrorAction SilentlyContinue
if (-not $Git) { throw "Git is required. Install Git for Windows and retry." }
$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("support-conditioned-vision-publish-" + [guid]::NewGuid().ToString("N"))
try {
    git clone $Repository $Work
    if ($LASTEXITCODE -ne 0) { throw "git clone failed; check repository URL and GitHub authentication." }
    robocopy $Source $Work /E /COPY:DAT /R:2 /W:2 /XD .git .venv __pycache__ .pytest_cache runs data /XF *.local.json *.pyc *.log
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE." }
    Push-Location $Work
    try {
        git add -A
        if ($LASTEXITCODE -ne 0) { throw "git add failed." }
        git diff --cached --quiet
        if ($LASTEXITCODE -gt 1) { throw "git diff failed." }
        if ($LASTEXITCODE -eq 1) {
            git commit -m "Publish Support-Conditioned Vision source"
            if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
        }
        git push origin main
        if ($LASTEXITCODE -ne 0) { throw "git push failed; resolve authentication or branch protection and retry." }
    } finally { Pop-Location }
    Write-Host "Published to $Repository"
} finally {
    if (Test-Path $Work) { Remove-Item $Work -Recurse -Force }
}
