@echo off 

set "currentver=11.7"
set "newversion=11.6"

setlocal EnableExtensions DisableDelayedExpansion

echo Updating all vcxproj files from %currentver% to %newversion%...

set "textFile=*.vcxproj"

set "rootDir=."

for /R "%rootDir%" %%j in ("%textFile%") do (
    for /f "delims=" %%i in ('type "%%~j" ^& break ^> "%%~j"') do (
        set "line=%%i"
        setlocal EnableDelayedExpansion
        set "line=!line:CUDA %currentver%.props=CUDA %newversion%.props!"
        set "line=!line:CUDA %currentver%.targets=CUDA %newversion%.targets!"
        >>"%%~j" echo(!line!
        endlocal
    )
    echo patched %%j
)

echo All done!

endlocal