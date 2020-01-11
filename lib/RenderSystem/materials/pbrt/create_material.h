/* create_material.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Glue between PBRT API and HostMaterial.
*/

#pragma once

// #include <system.h>
#include <rendersystem.h>

#include "paramset.h"
#include "pbrt_wrap.h"

#include <memory>

namespace pbrt
{
HostMaterial* CreateDisneyMaterial( const TextureParams& mp );
HostMaterial* CreateGlassMaterial( const TextureParams& mp );
HostMaterial* CreateMatteMaterial( const TextureParams& mp );
HostMaterial* CreateMetalMaterial( const TextureParams& mp );
HostMaterial* CreateMirrorMaterial( const TextureParams& mp );
HostMaterial* CreatePlasticMaterial( const TextureParams& mp );
HostMaterial* CreateSubstrateMaterial( const TextureParams& mp );
HostMaterial* CreateUberMaterial( const TextureParams& mp );
}; // namespace pbrt
