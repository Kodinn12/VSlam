# PowerShell script to setup stable SLAM environment
# Requirements: Conda (Miniconda/Anaconda) installed

$EnvName = "slam_stable"
$PythonVer = "3.10"

Write-Host "--- Initializing Stable SLAM Environment: $EnvName (Python $PythonVer) ---" -ForegroundColor Cyan

# 1. Create Conda environment
Write-Host "[1/4] Creating conda environment..." -ForegroundColor Green
conda create -n $EnvName python=$PythonVer -y

# 2. Activate environment and install dependencies
Write-Host "[2/4] Installing dependencies from stable_requirements.txt..." -ForegroundColor Green
conda run -n $EnvName pip install -r stable_requirements.txt

# 3. Verify installation
Write-Host "[3/4] Verifying GPU support..." -ForegroundColor Green
conda run -n $EnvName python -c "import torch; import cupy; print('PyTorch CUDA:', torch.cuda.is_available()); print('CuPy Available:', cupy.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 4. Final Instructions
Write-Host "--- Setup Complete! ---" -ForegroundColor Cyan
Write-Host "To use this environment, run:" -ForegroundColor Yellow
Write-Host "conda activate $EnvName" -ForegroundColor Yellow
Write-Host "python run_gpu.py" -ForegroundColor Yellow
