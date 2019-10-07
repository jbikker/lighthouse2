echo off

set OUT_DIR=%~dp0

IF NOT EXIST "%OUT_DIR%" (
	mkdir "%OUT_DIR%"
)

for /r %%i in (rt_*.comp) do %OUT_DIR%\..\..\vulkan\bin\glslangValidator -V "%%i" -o "%OUT_DIR%%%~ni%%~xi.spv"
for /r %%i in (rt_*.glsl) do %OUT_DIR%\..\..\vulkan\bin\glslangValidator -V "%%i" -o "%OUT_DIR%%%~ni%%~xi.spv"
for /r %%i in (rt_*.rchit) do %OUT_DIR%\..\..\vulkan\bin\glslangValidator -V "%%i" -o "%OUT_DIR%%%~ni%%~xi.spv"
for /r %%i in (rt_*.rahit) do %OUT_DIR%\..\..\vulkan\bin\glslangValidator -V "%%i" -o "%OUT_DIR%%%~ni%%~xi.spv"
for /r %%i in (rt_*.rmiss) do %OUT_DIR%\..\..\vulkan\bin\glslangValidator -V "%%i" -o "%OUT_DIR%%%~ni%%~xi.spv"
for /r %%i in (rt_*.vert) do %OUT_DIR%\..\..\vulkan\bin\glslangValidator -V "%%i" -o "%OUT_DIR%%%~ni%%~xi.spv"
for /r %%i in (rt_*.frag) do %OUT_DIR%\..\..\vulkan\bin\glslangValidator -V "%%i" -o "%OUT_DIR%%%~ni%%~xi.spv"

pause
