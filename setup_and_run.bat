@echo off
chcp 65001 >nul
title Raon Vision Search
setlocal enabledelayedexpansion

echo ============================================================
echo   Raon Vision Search - Full Auto Setup
echo ============================================================
echo.

REM -- 1. Find Python 3.10~3.12 via py launcher --
set PYTHON_CMD=

py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.12
    goto :found_python
)
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.11
    goto :found_python
)
py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.10
    goto :found_python
)

REM -- Python 3.10~3.12 not found: auto download and install --
echo.
echo ============================================================
echo   [AUTO] Python 3.10~3.12 not found.
echo   [AUTO] Downloading Python 3.12.10 installer...
echo ============================================================

set PY_INSTALLER=%TEMP%\python-3.12.10-amd64.exe
set PY_URL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe

echo [AUTO] Downloading from python.org ...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%'"
if %errorlevel% neq 0 (
    echo [ERROR] Download failed. Check internet connection.
    pause
    exit /b 1
)
echo [AUTO] Download complete: %PY_INSTALLER%

echo [AUTO] Installing Python 3.12.10 silently...
echo [AUTO] This may take 1~2 minutes. Please wait...
"%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=1 Include_test=0 Include_pip=1

REM Wait for installer process to fully finish
timeout /t 15 /nobreak >nul

REM Verify
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Auto-install failed.
    echo         Please install manually:
    echo         https://www.python.org/downloads/release/python-31210/
    pause
    exit /b 1
)

set PYTHON_CMD=py -3.12
echo [AUTO] Python 3.12.10 installed successfully.

REM Clean up installer file
del "%PY_INSTALLER%" >nul 2>&1

:found_python
for /f "tokens=2" %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% detected  [command: %PYTHON_CMD%]

REM -- 2. venv create / activate --
if not exist "%~dp0venv" (
    echo.
    echo [INFO] Creating virtual environment with %PYTHON_CMD% ...
    %PYTHON_CMD% -m venv "%~dp0venv"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo [OK] venv created
)

call "%~dp0venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate venv
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [STEP 1/4] Upgrade pip
echo ============================================================
python -m pip install --upgrade pip >nul 2>&1

echo.
echo ============================================================
echo   [STEP 2/4] Install PyTorch (CPU default)
echo ============================================================

set TORCH_OK=0

echo [INFO] Installing PyTorch CPU build...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if !errorlevel! equ 0 (
    set TORCH_OK=1
    echo [OK] PyTorch CPU build installed.
) else (
    echo.
    echo [ERROR] PyTorch CPU installation failed.
    pause
    exit /b 1
)

set GPU_TYPE=cpu

REM -- NVIDIA GPU 감지 --
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_TYPE=cuda
    for /f "tokens=*" %%d in ('nvidia-smi -L 2^>nul') do echo [GPU] %%d
    goto :gpu_found
)

REM -- AMD GPU 감지 (ROCm) --
rocm-smi >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_TYPE=rocm
    echo [GPU] AMD GPU detected via rocm-smi
    goto :gpu_found
)

REM -- Windows에서 AMD GPU 감지 (wmic) --
for /f "tokens=*" %%g in ('wmic path win32_videocontroller get name 2^>nul ^| findstr /i "Radeon"') do (
    set GPU_TYPE=rocm
    echo [GPU] AMD GPU detected: %%g
    goto :gpu_found
)

echo [GPU] No GPU detected. Using CPU build.
goto :install_torch

:gpu_found
echo [GPU] Detected type: %GPU_TYPE%

:install_torch
if "%GPU_TYPE%"=="cuda" (
    echo [INFO] Installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-deps
    if !errorlevel! equ 0 (
        echo [OK] PyTorch CUDA 12.1 installed.
    ) else (
        echo [WARN] CUDA install failed. Falling back to CPU.
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-deps
    )
) else if "%GPU_TYPE%"=="rocm" (
    if "%OS%"=="Windows_NT" (
        echo.
        echo [WARN] ROCm is NOT supported on Windows.
        echo [WARN] AMD GPU detected but using CPU build.
        echo [WARN] For ROCm acceleration, run on Linux.
        echo.
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-deps
    ) else (
        echo [INFO] Installing PyTorch with ROCm 6.2...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2 --no-deps
        if !errorlevel! equ 0 (
            echo [OK] PyTorch ROCm 6.2 installed.
        ) else (
            echo [WARN] ROCm install failed. Falling back to CPU.
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-deps
        )
    )
) else (
    echo [INFO] Installing PyTorch CPU build...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-deps
)

python -c "import torch; hip = getattr(torch.version, 'hip', None); mode = 'ROCm' if hip else ('CUDA' if torch.cuda.is_available() else 'CPU'); print('[VERIFY] PyTorch:', torch.__version__, '| Accelerator:', mode, '| Available:', torch.cuda.is_available())"

echo.
echo ============================================================
echo   [STEP 3/4] Install remaining packages
echo ============================================================

pip install -r "%~dp0requirements.txt"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Package installation failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [STEP 4/4] Verify installation
echo ============================================================
python -c "import torch; print('  PyTorch     :', torch.__version__); print('  CUDA avail  :', torch.cuda.is_available())"
python -c "import timm; print('  timm        :', timm.__version__)"
python -c "import transformers; print('  transformers:', transformers.__version__)"
python -c "import webview; print('  pywebview   : OK')"
python -c "import safetensors; print('  safetensors : OK')"

echo.
echo ============================================================
echo   All done! Starting app...
echo ============================================================
echo.

python "%~dp0app.py"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed
    pause
)

endlocal