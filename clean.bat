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

rd lib\RenderCore_Minimal\lib /S /Q
rd lib\RenderCore_Minimal\x64 /S /Q
rd lib\RenderCore_PrimeRef\lib /S /Q
rd lib\RenderCore_PrimeRef\x64 /S /Q
rd lib\RenderCore_OptixPrime_B\lib /S /Q
rd lib\RenderCore_OptixPrime_B\x64 /S /Q
rd lib\RenderCore_OptixRTX_B\lib /S /Q
rd lib\RenderCore_OptixRTX_B\x64 /S /Q
del lib\RenderCore_OptixRTX_B\optix\*.ptx /S /Q
rd lib\RenderCore_OptixPrime_BDPT\lib /S /Q
rd lib\RenderCore_OptixPrime_BDPT\x64 /S /Q
rd lib\RenderCore_OptixPrime_PBRT\lib /S /Q
rd lib\RenderCore_OptixPrime_PBRT\x64 /S /Q
del lib\RenderCore_OptixPrime_BDPT\optix\*.ptx /S /Q
rd lib\RenderCore_Optix7\lib /S /Q
rd lib\RenderCore_Optix7\x64 /S /Q
rd lib\RenderCore_Optix7Guiding\lib /S /Q
rd lib\RenderCore_Optix7Guiding\x64 /S /Q
del lib\RenderCore_Optix7\optix\*.ptx /S /Q
rd lib\RenderCore_Optix7Filter\lib /S /Q
rd lib\RenderCore_Optix7Filter\x64 /S /Q
del lib\RenderCore_Optix7Filter\optix\*.ptx /S /Q
rd lib\RenderCore_SoftRasterizer\lib /S /Q
rd lib\RenderCore_SoftRasterizer\x64 /S /Q
rd lib\RenderCore_OpenCL\lib /S /Q
rd lib\RenderCore_OpenCL\x64 /S /Q
rd lib\RenderCore_Embree\x64 /S /Q
rd lib\RenderCore_Embree\lib /S /Q
rd lib\RenderCore_Vulkan_RT\lib /S /Q
rd lib\RenderCore_Vulkan_RT\x64 /S /Q
rd coredlls\debug /S /Q
rd coredlls\release /S /Q

rem | Clean up applications

rd apps\ai_debugger\x64 /S /Q
del apps\ai_debugger\data\textures\*.bin
del apps\ai_debugger\data\sky_15.bin
del apps\ai_debugger\*.exe /Q
del apps\ai_debugger\*.iobj /Q
del apps\ai_debugger\*.ipdb /Q
del apps\ai_debugger\*.pdb /Q
del apps\ai_debugger\*.ilk /Q
del apps\ai_debugger\*.exp /Q
del apps\ai_debugger\*.lib /Q
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
del apps\tinyapp\*.exp /Q
del apps\tinyapp\*.lib /Q
rd apps\geneticapp\x64 /S /Q
del apps\geneticapp\data\textures\*.bin
del apps\geneticapp\data\sky_15.bin
del apps\geneticapp\*.exe /Q
del apps\geneticapp\*.iobj /Q
del apps\geneticapp\*.ipdb /Q
del apps\geneticapp\*.pdb /Q
del apps\geneticapp\*.ilk /Q
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
del apps\imguiapp\*.exp /Q
del apps\imguiapp\*.lib /Q
rd apps\benchmarkapp\x64 /S /Q
del apps\benchmarkapp\data\textures\*.bin
del apps\benchmarkapp\data\mattest\textures\*.bin
del apps\benchmarkapp\data\sky_15.bin
del apps\benchmarkapp\*.exe /Q
del apps\benchmarkapp\*.iobj /Q
del apps\benchmarkapp\*.ipdb /Q
del apps\benchmarkapp\*.pdb /Q
del apps\benchmarkapp\*.ilk /Q
del apps\benchmarkapp\*.exp /Q
del apps\benchmarkapp\*.lib /Q
rd apps\rtxbvhreveng\x64 /S /Q
del apps\rtxbvhreveng\data\sky_15.bin
del apps\rtxbvhreveng\*.exe /Q
del apps\rtxbvhreveng\*.iobj /Q
del apps\rtxbvhreveng\*.ipdb /Q
del apps\rtxbvhreveng\*.pdb /Q
del apps\rtxbvhreveng\*.ilk /Q
del apps\rtxbvhreveng\*.exp /Q
del apps\rtxbvhreveng\*.lib /Q
rd apps\tinyapplua\x64 /S /Q
del apps\tinyapplua\data\sky_15.bin
del apps\tinyapplua\*.exe /Q
del apps\tinyapplua\*.iobj /Q
del apps\tinyapplua\*.ipdb /Q
del apps\tinyapplua\*.pdb /Q
del apps\tinyapplua\*.ilk /Q
del apps\tinyapplua\*.exp /Q
del apps\tinyapplua\*.lib /Q
rd apps\pbrtdemoapp\x64 /S /Q
del apps\pbrtdemoapp\data\textures\*.bin
del apps\pbrtdemoapp\data\mattest\textures\*.bin
del apps\pbrtdemoapp\data\sky_15.bin
del apps\pbrtdemoapp\*.exe /Q
del apps\pbrtdemoapp\*.iobj /Q
del apps\pbrtdemoapp\*.ipdb /Q
del apps\pbrtdemoapp\*.pdb /Q
del apps\pbrtdemoapp\*.ilk /Q
del apps\pbrtdemoapp\*.exp /Q
del apps\pbrtdemoapp\*.lib /Q

rem | Clean up other components

rd lib\RenderSystem\lib /S /Q
rd lib\RenderSystem\x64 /S /Q
rd lib\platform\lib /S /Q
rd lib\platform\x64 /S /Q