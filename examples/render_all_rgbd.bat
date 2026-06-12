@echo off
rem Render every Models\*.obj from one monocular viewpoint into out\<name>_az<az>_el<el>\
rem (rgb.png, depth.npy, cloud_cam.ply, cloud_world.ply, camera.json).
rem
rem Usage:  examples\render_all_rgbd.bat [azimuth] [elevation]
rem         (defaults: azimuth 40, elevation 15 - same as the knotty example)
rem
rem Uses the open3d_render conda env's python; override with:
rem         set PYTHON_RGBD=C:\path\to\python.exe
setlocal

if "%PYTHON_RGBD%"=="" set "PYTHON_RGBD=C:\Users\admin\anaconda3\envs\open3d_render\python.exe"

set "AZ=40"
set "EL=15"
if not "%~1"=="" set "AZ=%~1"
if not "%~2"=="" set "EL=%~2"

cd /d "%~dp0.."

set FAILED=
for %%F in (Models\*.obj) do (
    echo === %%~nxF ===
    "%PYTHON_RGBD%" examples\render_rgbd.py "%%F" --azimuth %AZ% --elevation %EL%
    if errorlevel 1 (
        echo FAILED: %%~nxF
        set "FAILED=1"
    )
)

if defined FAILED (
    echo.
    echo Some models failed - see FAILED lines above.
    exit /b 1
)
echo.
echo All models rendered into out\
endlocal
