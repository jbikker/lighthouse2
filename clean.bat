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

rd lib\RenderCore_OptixPrime_B\lib /S /Q
rd lib\RenderCore_OptixPrime_B\x64 /S /Q
rd lib\RenderCore_OptixRTX_B\lib /S /Q
rd lib\RenderCore_OptixRTX_B\x64 /S /Q
del lib\RenderCore_OptixRTX_B\optix\*.ptx /S /Q
rd lib\RenderCore_SoftRasterizer\lib /S /Q
rd lib\RenderCore_SoftRasterizer\x64 /S /Q
rd coredlls\debug /S /Q
rd coredlls\release /S /Q

rem | Clean up applications

rd apps\tinyapp\x64 /S /Q
del apps\tinyapp\data\textures\*.bin
del apps\tinyapp\data\sky_15.bin
del apps\tinyapp\*.exe /Q
del apps\tinyapp\*.iobj /Q
del apps\tinyapp\*.ipdb /Q
del apps\tinyapp\*.pdb /Q
del apps\tinyapp\*.ilk /Q

rem | Clean up other components

rd lib\RenderSystem\lib /S /Q
rd lib\RenderSystem\x64 /S /Q
rd lib\platform\lib /S /Q
rd lib\platform\x64 /S /Q