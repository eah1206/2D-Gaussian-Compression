@echo off
setlocal enabledelayedexpansion

:: ─── Configuration ────────────────────────────────────────────────────────────
set PYTHON_SCRIPT=Claude2DGauss.py
set FOLDER=Office_Confrontation
set TYPE=residuals
set OTHER_TYPE=frames
set STEPS=1000
set MAX_INDEX=192
set BATCH_SIZE=16

:: ─── Define the n_gaussians values to sweep over ──────────────────────────────
set N_GAUSSIANS_LIST=10 100

:: ─── Run Loop ─────────────────────────────────────────────────────────────────
for %%N in (%N_GAUSSIANS_LIST%) do (
    echo ==========================================
    echo  Running with n_gaussians=%%N for %TYPE%
    echo  Started at: !date! !time!
    echo ==========================================

    python %PYTHON_SCRIPT% ^
        --folder %FOLDER% ^
        --type %TYPE% ^
        --n_gaussians %%N ^
        --steps %STEPS% ^
        --max_index %MAX_INDEX% ^
        --batch_size %BATCH_SIZE%

    if errorlevel 1 (
        echo Run with n_gaussians=%%N failed. Stopping script.
        exit /b 1
    )

    echo  Finished at: !date! !time!
    echo  Run with n_gaussians=%%N for %TYPE% completed successfully.
    echo.

    if %TYPE%==frames (
        set OTHER_TYPE=residuals
        echo !OTHER_TYPE!
    )

    echo ==========================================
    echo  Running with n_gaussians=%%N for %OTHER_TYPE%
    echo  Started at: !date! !time!
    echo ==========================================

    python %PYTHON_SCRIPT% ^
        --folder %FOLDER% ^
        --type %OTHER_TYPE% ^
        --n_gaussians %%N ^
        --steps %STEPS% ^
        --max_index %MAX_INDEX% ^
        --batch_size %BATCH_SIZE%

    if errorlevel 1 (
        echo Run with n_gaussians=%%N failed. Stopping script.
        exit /b 1
    )

    echo  Finished at: !date! !time!
    echo  Run with n_gaussians=%%N for %OTHER_TYPE% completed successfully.
    echo.
)

echo All runs completed.