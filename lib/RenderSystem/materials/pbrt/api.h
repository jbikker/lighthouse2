/* api.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Declarations for the PBRT API.
*/

#if defined( _MSC_VER )
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_API_H
#define PBRT_CORE_API_H

#include "pbrt_wrap.h"

#include <rendersystem.h>

namespace pbrt
{

// API Function Declarations
void pbrtInit( const Options& opt, HostScene* hs );
void pbrtCleanup();
void pbrtIdentity();
void pbrtTranslate( Float dx, Float dy, Float dz );
void pbrtRotate( Float angle, Float ax, Float ay, Float az );
void pbrtScale( Float sx, Float sy, Float sz );
void pbrtLookAt( Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
				 Float ux, Float uy, Float uz );
void pbrtConcatTransform( Float transform[16] );
void pbrtTransform( Float transform[16] );
void pbrtCoordinateSystem( const std::string& );
void pbrtCoordSysTransform( const std::string& );
void pbrtActiveTransformAll();
void pbrtActiveTransformEndTime();
void pbrtActiveTransformStartTime();
void pbrtTransformTimes( Float start, Float end );
void pbrtPixelFilter( const std::string& name, const ParamSet& params );
void pbrtFilm( const std::string& type, const ParamSet& params );
void pbrtSampler( const std::string& name, const ParamSet& params );
void pbrtAccelerator( const std::string& name, const ParamSet& params );
void pbrtIntegrator( const std::string& name, const ParamSet& params );
void pbrtCamera( const std::string&, const ParamSet& cameraParams );
void pbrtMakeNamedMedium( const std::string& name, const ParamSet& params );
void pbrtMediumInterface( const std::string& insideName,
						  const std::string& outsideName );
void pbrtWorldBegin();
void pbrtAttributeBegin();
void pbrtAttributeEnd();
void pbrtTransformBegin();
void pbrtTransformEnd();
void pbrtTexture( const std::string& name, const std::string& type,
				  const std::string& texname, const ParamSet& params );
void pbrtMaterial( const std::string& name, const ParamSet& params );
void pbrtMakeNamedMaterial( const std::string& name, const ParamSet& params );
void pbrtNamedMaterial( const std::string& name );
void pbrtLightSource( const std::string& name, const ParamSet& params );
void pbrtAreaLightSource( const std::string& name, const ParamSet& params );
void pbrtShape( const std::string& name, const ParamSet& params );
void pbrtReverseOrientation();
void pbrtObjectBegin( const std::string& name );
void pbrtObjectEnd();
void pbrtObjectInstance( const std::string& name );
void pbrtWorldEnd();

void pbrtParseFile( std::string filename );
void pbrtParseString( std::string str );

}; // namespace pbrt

#endif // PBRT_CORE_API_H
