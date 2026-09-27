<#
.SYNOPSIS
    Creates .venv (Python 3.14) and installs both projects with the right PyTorch build.

.DESCRIPTION
    1. Checks that the Python install manager ("py") has Python 3.14.
    2. Creates the virtual environment .venv in the repository root.
    3. Picks a PyTorch build: detects an NVIDIA GPU with nvidia-smi and chooses
       cu130 (RTX 20-series or newer with driver 580+), cu126 (older GPU or
       driver), or cpu. Override with -Torch.
    4. Installs torch + torchvision from PyTorch's own wheel index (PyPI only
       has CPU builds for Windows), then everything in requirements.txt.
    5. Registers nbstripout as a git filter and runs scripts/check_env.py.

    Safe to re-run. See docs/02_INSTALLATION.md for what each step does.

.PARAMETER Torch
    auto (default), cpu, cu130 or cu126.

.PARAMETER PythonVersion
    Python version for the venv (default 3.14).

.PARAMETER Recreate
    Delete and rebuild .venv from scratch.

.EXAMPLE
    .\scripts\setup_env.ps1
.EXAMPLE
    .\scripts\setup_env.ps1 -Torch cpu -Recreate
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cu130', 'cu126')]
    [string]$Torch = 'auto',
    [string]$PythonVersion = '3.14',
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo   # requirements.txt uses paths relative to the repo root

function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Arguments -join ' ')" }
}

function Get-NvidiaGpu {
    # Returns @{ Name; DriverMajor; ComputeCap } or $null. Never throws.
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Capture all output first (piping into Select-Object -First can cut the process short).
        $output = @(& nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>$null)
        $computeCap = $null
        if ($LASTEXITCODE -ne 0 -or $output.Count -eq 0) {
            # Very old drivers do not know compute_cap; ask for the rest.
            $output = @(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null)
            if ($LASTEXITCODE -ne 0 -or $output.Count -eq 0) { return $null }
        }
        $parts = "$($output[0])" -split ',\s*'
        if ($parts.Count -ge 3) {
            $computeCap = [double]::Parse($parts[2], [Globalization.CultureInfo]::InvariantCulture)
        }
        return @{ Name = $parts[0]; DriverMajor = [int]($parts[1].Split('.')[0]); ComputeCap = $computeCap }
    }
    catch { return $null }
    finally { $ErrorActionPreference = $previous }
}

Write-Host "`n== 1/5 Python $PythonVersion" -ForegroundColor Cyan
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The 'py' command was not found. Install the Python install manager first (docs/02_INSTALLATION.md, step 2)."
}
& py "-$PythonVersion" --version
if ($LASTEXITCODE -ne 0) {
    throw "Python $PythonVersion is not installed. Run:  py install $PythonVersion   then re-run this script."
}

Write-Host "`n== 2/5 Virtual environment (.venv)" -ForegroundColor Cyan
if ($Recreate -and (Test-Path .venv)) {
    Write-Host 'Removing existing .venv'
    Remove-Item -Recurse -Force .venv
}
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Invoke-Native 'py' @("-$PythonVersion", '-m', 'venv', '.venv')
}
else {
    Write-Host '.venv already exists (use -Recreate to rebuild it)'
}
Invoke-Native $python @('-m', 'pip', 'install', '--upgrade', 'pip')

Write-Host "`n== 3/5 PyTorch build" -ForegroundColor Cyan
if ($Torch -eq 'auto') {
    $Torch = 'cpu'
    $gpu = Get-NvidiaGpu
    if ($null -eq $gpu) {
        Write-Host 'No NVIDIA GPU detected (or no driver): using the CPU build.'
    }
    else {
        Write-Host "Found $($gpu.Name), driver $($gpu.DriverMajor), compute capability $($gpu.ComputeCap)"
        if ($null -ne $gpu.ComputeCap -and $gpu.ComputeCap -ge 7.5 -and $gpu.DriverMajor -ge 580) {
            $Torch = 'cu130'
        }
        elseif ($gpu.DriverMajor -ge 528) {
            $Torch = 'cu126'
            if ($null -ne $gpu.ComputeCap -and $gpu.ComputeCap -ge 7.5) {
                Write-Host 'Tip: update the NVIDIA driver to 580+ and re-run with -Recreate to get the newer CUDA 13 build.' -ForegroundColor Yellow
            }
        }
        else {
            Write-Host 'Driver is too old for any current PyTorch CUDA build: using CPU. Update the driver (docs/02_INSTALLATION.md, step 5).' -ForegroundColor Yellow
        }
    }
}
Write-Host "Selected PyTorch build: $Torch"

# Read the pinned versions from the requirements file so they live in one place.
$pins = Get-Content 'requirements\ribbon_classifier.txt'
$torchVersion = ($pins | Select-String '^torch==([^\s#]+)').Matches[0].Groups[1].Value
$visionVersion = ($pins | Select-String '^torchvision==([^\s#]+)').Matches[0].Groups[1].Value
Invoke-Native $python @('-m', 'pip', 'install', "torch==$torchVersion", "torchvision==$visionVersion",
    '--index-url', "https://download.pytorch.org/whl/$Torch")

Write-Host "`n== 4/5 Project packages (requirements.txt)" -ForegroundColor Cyan
Invoke-Native $python @('-m', 'pip', 'install', '-r', 'requirements.txt')

Write-Host "`n== 5/5 Git filter + environment check" -ForegroundColor Cyan
if (Get-Command git -ErrorAction SilentlyContinue) {
    Invoke-Native (Join-Path $repo '.venv\Scripts\nbstripout.exe') @('--install')
}
else {
    Write-Host 'git not found: skipping nbstripout (notebook outputs would be committed).' -ForegroundColor Yellow
}
Invoke-Native $python @('scripts\check_env.py')

Write-Host "`nDone. In each new terminal, activate the environment with:" -ForegroundColor Green
Write-Host '    .\.venv\Scripts\Activate.ps1'
