rem | This file deletes all files that can be regenerated with 
rem | a rebuild / run in VS2017. Execute this when archiving, 
rem | handing in assignments etc.
rem | Make sure VS2017 is *closed* for best results.

del *.suo /Q
del *.suo /AH /Q
del *.sdf /Q
rd ipch /S /Q
rd .vs /S /Q

rem | Clean up RenderCore folders

rd lib\RenderCore_OptixPrime\lib /S /Q
rd lib\RenderCore_OptixPrime\x64 /S /Q
rd lib\RenderCore_PrimeRef\lib /S /Q
rd lib\RenderCore_PrimeRef\x64 /S /Q
rd lib\RenderCore_OptixPrime_B\lib /S /Q
rd lib\RenderCore_OptixPrime_B\x64 /S /Q
rd lib\RenderCore_OptixRTX\lib /S /Q
rd lib\RenderCore_OptixRTX\x64 /S /Q
del lib\RenderCore_OptixRTX\optix\*.ptx /S /Q
rd lib\RenderCore_OptixRTX_B\lib /S /Q
rd lib\RenderCore_OptixRTX_B\x64 /S /Q
del lib\RenderCore_OptixRTX_B\optix\*.ptx /S /Q
rd lib\RenderCore_Optix7\lib /S /Q
rd lib\RenderCore_Optix7\x64 /S /Q
del lib\RenderCore_Optix7\optix\*.ptx /S /Q
rd lib\RenderCore_SoftRasterizer\lib /S /Q
rd lib\RenderCore_SoftRasterizer\x64 /S /Q
rd lib\RenderCore_RTX_AO\lib /S /Q
rd lib\RenderCore_RTX_AO\x64 /S /Q
del lib\RenderCore_RTX_AO\optix\*.ptx /S /Q
rd lib\RenderCore_OpenCL\lib /S /Q
rd lib\RenderCore_OpenCL\x64 /S /Q
rd lib\RenderCore_Embree\x64 /S /Q
rd lib\RenderCore_Embree\lib /S /Q
rd coredlls\debug /S /Q
rd coredlls\release /S /Q

rem | Clean up applications

rd apps\basicapp\x64 /S /Q
del apps\basicapp\data\textures\*.bin
del apps\basicapp\data\sky_15.bin
del apps\basicapp\*.exe /Q
del apps\basicapp\*.iobj /Q
del apps\basicapp\*.ipdb /Q
del apps\basicapp\*.pdb /Q
del apps\basicapp\*.ilk /Q
rd apps\tinyapp\x64 /S /Q
del apps\tinyapp\data\textures\*.bin
del apps\tinyapp\data\sky_15.bin
del apps\tinyapp\*.exe /Q
del apps\tinyapp\*.iobj /Q
del apps\tinyapp\*.ipdb /Q
del apps\tinyapp\*.pdb /Q
del apps\tinyapp\*.ilk /Q
rd apps\app_matui\x64 /S /Q
del apps\app_matui\data\textures\*.bin
del apps\app_matui\data\mattest\textures\*.bin
del apps\app_matui\data\sky_15.bin
del apps\app_matui\*.exe /Q
del apps\app_matui\*.iobj /Q
del apps\app_matui\*.ipdb /Q
del apps\app_matui\*.pdb /Q
del apps\app_matui\*.ilk /Q
rd apps\imguiapp\x64 /S /Q
del apps\imguiapp\data\textures\*.bin
del apps\imguiapp\data\mattest\textures\*.bin
del apps\imguiapp\data\sky_15.bin
del apps\imguiapp\*.exe /Q
del apps\imguiapp\*.iobj /Q
del apps\imguiapp\*.ipdb /Q
del apps\imguiapp\*.pdb /Q
del apps\imguiapp\*.ilk /Q

rem | Clean up other components

rd lib\RenderSystem\lib /S /Q
rd lib\RenderSystem\x64 /S /Q
rd lib\platform\lib /S /Q
rd lib\platform\x64 /S /Q