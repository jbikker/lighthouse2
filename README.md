# lighthouse2
Lighthouse 2 framework for real-time ray tracing

This is the public repo for Lighthouse 2, a rendering framework for real-time ray tracing / path tracing experiments. 
Lighthouse 2 uses a state-of-the-art wavefront / streaming ray tracing implementation to reach high ray througput on RTX hardware 
(using Optix 7) and pre-RTX hardware (using Optix 5 Prime) and soon on AMD hardware (using RadeonRays / OpenCL) and CPUs (using Embree).
A software rasterizer is also included, mostly as an example of a minimal API implementation.

![ScreenShot](/screenshots/lighthouse_cobra.png)

Quick pointers / Important advice:

* Building Lighthouse 2: CUDA 10 currently does *not* properly support vs2019; use vs2017 for now.
* Lighthouse 2 wiki: https://github.com/jbikker/lighthouse2/wiki (early stages)
* Trouble shooting page on the wiki: https://github.com/jbikker/lighthouse2/wiki/TroubleShooting
* Lighthouse 2 forum: https://ompf2.com/viewforum.php?f=18
* Follow the project on Twitter: @j_bikker

Lighthouse 2 uses a highly modular approach to ease the development of renderers.

The main layers are:

1. The application layer, which implements application logic and handles user input;
2. The RenderSystem, which handles scene I/O and host-side scene storage;
3. The render cores, which implement low-level rendering functionality.

Render cores have a common interface and are supplied to the RenderSystem as dlls. The RenderSystem supplies the cores with scene data 
(meshes, instances, triangles, textures, materials, lights) and sparse updates to this data.

The Lighthouse 2 project has the following target audience:

*Researchers*

Lighthouse 2 is designed to be a high-performance starting point for novel algorithms involving real-time ray tracing. This may include
new work on filtering, sampling, materials and lights. The provided ray tracers easily reach hundreds of millions of rays per second 
on NVidia and AMD GPUs. Combined with a generic GPGPU implementation, this enables a high level of freedom in the implementation of 
new code.

*Educators*

The Lighthouse 2 system implements all the boring things such as scene I/O, window management, user interfaces and access to ray tracing
APIs such as Optix, RadeonRays and Embree; your students can get straight to the interesting bits. The architecture of Lighthouse 2 is
carefully designed to be easily accessible. Very fast scene loading and carefully tuned project files ensure quick development cycles.

*Industry*

Lighthouse 2 is an R&D platform. It is however distributed with the Apache 2.0 license, which allows you to use the code in your
own products. Experimental cores can be shared with the community in binary / closed form, and application development is separated
from core development.

<b>What it is not</b>

The ray tracing infrastructure (with related scene management acceleration structure maintenance) should be close to optimal. The implemented estimators however (unidirectional path tracers without filtering and blue noise) are not, and neither is the shading
model (Lambert + speculars). This may or may not change depending on the use cases encountered. This video shows what can be
achieved with the platform: https://youtu.be/uEDTtu2ky3o .

Lighthouse 2 should compile out-of-the-box on Windows using Visual Studio 2017 / 2019. For the CUDA/Optix based cores CUDA 10.2 is required:

https://developer.nvidia.com/cuda-downloads

Optix 5.x, 6.0 and 7.0 libraries are included in the Lighthouse 2 download.

For more information on Lighthouse 2 please visit: http://jacco.ompf2.com.

<b>Credits</b>

Lighthouse 2 was developed at the Utrecht University, The Netherlands.

Lighthouse 2 uses the following libraries:<br>
Dear ImGui https://github.com/ocornut/imgui<br>
FreeImage http://freeimage.sourceforge.net<br>
Glad https://glad.dav1d.de<br>
GLFW https://www.glfw.org<br>
half 1.12 http://half.sourceforge.net<br>
tinygltf https://github.com/syoyo/tinygltf<br>
tinyobj https://github.com/syoyo/tinyobjloader<br>
tinyxml2 https://github.com/leethomason/tinyxml2<br>
zlib https://www.zlib.net

<b>Contributions</b>

* The Lighthouse2 Vulkan core (and sharedBSDF) was developed by Mèir Noordermeer (https://github.com/MeirBon).
* A Linux port by Marijn Suijten (https://github.com/MarijnS95) is being incorporated in the main repo.
* Animation code uses low-level optimizations by Alysha Bogaers and Naraenda Prasetya.
* OptixPrime_BDPT core by Guowei (Peter) Lu (https://github.com/pasu).

<b>Previous Work</b>

Lighthouse 2 implements research by (very incomplete):

* Marsaglia: random numbers
* Van Antwerpen, Laine, Karras, Aila: streaming path tracing
* Aila, Laine: persistent kernels
* Schied et al.: Spatiotemporal Variance-Guided Filtering (SVGF)
* Victor Voorhuis: improved SVGF for specular and glossy reprojection
* Eric Heitz: Blue noise distributions
