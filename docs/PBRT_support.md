# Support for PBRT scenes
Lighthouse 2 supports loading and rendering PBRT-v3 scene files. This includes the advanced materials powering these.

## BSDFs
PBRT implements all materials as a list of BxDFs, of which one is chosen at random to determine the reflected/refracted direction. All BxDFs are evaluated to retrieve the color value and the value of the PDF for that particular set of in and out directions.
10 of the 11 `BxDF` subclasses are implemented (with the exception of `FourierBSDF`, which is uses measured BSDF data that needs to be pushed to the GPU) and instantiated as part of a material where necessary. These implementations are carbon copies of the PBRT-v3 implementations with slight modifications to account for different types and available functions on the GPU. It is best to read [the PBRT book](http://www.pbr-book.org/3ed-2018/Reflection_Models/Basic_Interface.html) for more details.

## Material interface
The material interface is a similar pure virtual class that exposes querying functionality to a pathtracer implementation. These functions are tailored to current Lighthouse2 functionality. Subclasses of this interface implement the previously mentioned list of BxDFs (`BSDFStackMaterial`) and the original disney material. All PBRT materials implement a single method `ComputeScatteringFunctions` that convert a material description (tweakable material parameters as listed [here](https://www.pbrt.org/fileformat-v3.html#materials)) to a list of BxDFs.
There is also a Disney material (called `DisneyGltf`) that converts the original Lighthouse2 description to a list of BxDFs.

## Efficiency on the GPU
It is slow to allocate BxDF objects and materials on the fly (especially on the GPU), for which reason templates are used to create a compile-time known storage type. This allows the compiler and/or kernel invocation to allocate enough registers or memory beforehand, saving expensive memory allocation at the expense of overallocating for objects that may never need to be fitted.

Virtual functions are used in favour of switch-case tables, to make the code more readable and close to PBRT, which is the ultimate goal. While perhaps not as fast as a plain switch-case (this is to be confirmed), it allows pretty much direct copy-pasting of new (or yet-unimplemented) materials and BxDFs from PBRT and other sources. Grouping related functions in a class like this instead of spreading over a bunch of switch-case statements across the code greatly aids in maintainability.

## Scenes
Ready-to-use PBRT scenes can be pulled from [the official website](https://pbrt.org/scenes-v3.html), which also reference a small repository of scenes from [Benedikt Bitterli](https://benedikt-bitterli.me/resources/).

While efficiency has just been discussed above, these scenes still render much, much faster even on older/slower graphics cards. When manually rendering these scenes with `pbrt`, expect much longer rendering times, with a worse (read: more noisy) result. The desired sample count seems to have been increased when generating the images on the website, which is not reflected in the scene files.

### Coffee Maker
![Coffee Maker](/screenshots/coffee.png)
Original:
[![Original](https://benedikt-bitterli.me/resources/thumb/coffee.png)](https://benedikt-bitterli.me/resources/images/coffee.png)
Notice the bright reflection on the left side of the handle, and the black plastic that appears brighter and less specular than the original.
PBRT doesn't generate an image with this deep orange color. It seems to have been graded afterwards.
PBRT does however have a reflection on the inside of the metal ring, which is missing on the other pictures. This has likely been a global switch to double-sided surfaces?
There is also some fringing below the brand name on the black button, and across the center line where the pot connects to the top.

### Salle de bain
TODO: Redo with more bounces, sinks and wood is too dark!
![Salle de bain](/screenshots/bathroom2.png)
Original:
[![Original](https://benedikt-bitterli.me/resources/thumb/bathroom2.png)](https://benedikt-bitterli.me/resources/images/bathroom2.png)
The area light in this scene appears more yellow, and overexposes everything around it.

### Glass of water
![Glass of water](/screenshots/glass-of-water.png)
Original:
[![Original](https://benedikt-bitterli.me/resources/thumb/glass-of-water.png)](https://benedikt-bitterli.me/resources/images/glass-of-water.png)
Notice that this scene is also less dark than the original. There are less shadows or at least not as pronounced, and they are completely absent under the puddles around the melting icecubes.

### Lamp
This lamp is modeled with >2000 area lights representing the lightbulb.
![Lamp](/screenshots/lamp.png)
[![Original](https://benedikt-bitterli.me/resources/thumb/lamp.png)](https://benedikt-bitterli.me/resources/images/lamp.png)
Notice that the background is darker than the original (hiding the shadows, especially those cast by the head and arealights, which are blocking the distant light source), and the matte black is way less shiny and dark just like the Coffee Maker. The specular spot from the distant light source is completely absent.
The back area light on the left side of the picture casts a hard shadow.

## What's missing?
- Parsing simple and advanced configuration nodes from PBRT files; look for `not implemented` markers in [api.cpp](../lib/RenderSystem/materials/pbrt/api.cpp) for the current status.
  For example:
  - Variable max path length: certain scenes with lots of diffuse bounces or transmission have higher requirements than what is set by default.
- Support for participating media (`MakeNamedMedium` and `MediumInterface`).
- This implementation supports rendering `RGBSpectrum` (as `float3`) only. `SampledSpectrum`s (describing intensity for a number of light frquency bins) is not supported.
- Not all materials are supported. Complicated ones such as `Hair` require a lot of extra code, while others should be relatively easy to add when necessary for a scene.
- Bumpmapping is not implemented yet.

Note there are a bunch of `TODO` comments spread across the code. These range from simple items to those requiring major architectural changes to the project, which have been deemed out of scope for this implementation.

## Bugs
- Rotation and/or transformed skydome loading goes the wrong way around an axis
- Uber material Ks on the `chopper-titan` scene looks lighter than it should be
- Window slits in the `dining-room` scene do not cast proper shadows on the wall
  - Can be skydome transformation and the distant light source

## Windows bugs
- CUDA Kernel assertions do not compile in debug mode (except with #define NDEBUG, defeating the purpose).
- Compile-time sanity checks are disabled thanks to broken templated type aliases.
- CoreMaterialDesc too large due to alignment.
