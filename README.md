# lighthouse2
Lighthouse 2 framework for real-time ray tracing

This is the public repo for Lighthouse 2, a rendering framework for real-time ray tracing / path tracing experiments. 
Lighthouse 2 uses a state-of-the-art wavefront / streaming ray tracing implementation to reach high ray througput on RTX hardware 
(using Optix) and pre-RTX hardware (using Optix Prime) and soon on AMD hardware (using Radeon Rays) and CPUs (using Embree).
A software rasterizer is also included, mostly as an example of a minimal API implementation.

Lighthouse 2 uses a highly modular approach to ease the development of renderers.

The main layers are:

1. The application layer, which implements application logic and handles user input;
2. The RenderSystem, which handles scene I/O and host-side scene storage;
3. The render cores, which implement low-level rendering functionality.

Render cores have a common interface and are supplied to the RenderSystem as dlls. The RenderSystem supplies the cores with scene data 
(meshes, instances, triangles, textures, materials, lights) and sparse updates to this data.

The Lighthouse 2 project has the following target audience:

Researchers

Lighthouse 2 can serve as a high-performance starting point for novel algorithms involving real-time ray tracing. This may include
new work on filtering, sampling, materials and lights. The provided ray tracers easily reach hundreds of millions of rays per second 
on NVidia and AMD GPUs. Combined with a generic GPGPU implementation, this enables a high level of freedom in the implementation of 
new code.

Educators

The Lighthouse 2 system implements all the boring things such as scene I/O, window management, user interfaces and access to ray tracing
APIs such as Optix, RadeonRays and Embree. Students can get straight to the interesting bits. The architecture of Lighthouse 2 is
carefully designed to be easily accessible.

Industry

Lighthouse 2 is an R&D platform. It is however distributed with the Apache 2.0 license, which allows you to use the code in your
own products. Experimental cores can be shared with the community in binary / closed form, and application development is separated
from core development.

For more information on Lighthouse 2 please visit
http://jacco.ompf2.com
