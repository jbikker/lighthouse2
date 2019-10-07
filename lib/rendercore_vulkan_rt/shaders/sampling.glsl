/* sampling.glsl - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef SAMPLING_H
#define SAMPLING_H

#include "../bindings.h"

layout( set = 0, binding = cTEXTURE_ARGB32 ) buffer ARGB32Buffer { uint pixels[]; } ARGB32Data;
layout( set = 0, binding = cTEXTURE_ARGB128 ) buffer ARGB128Buffer { vec4 pixels[]; } ARGB128Data;
layout( set = 0, binding = cTEXTURE_NRM32 ) buffer NRM32Buffer { uint pixels[]; } NRM32Data;

vec4 uintToVec4( const uint v4 )
{
	const float r = 1.0f / 256.0f;
	return vec4( float(v4 & 255) * r, float((v4 >> 8) & 255) * r, float((v4 >> 16) & 255) * r, float(v4 >> 24) * r );
}

vec4 FetchTexel(
	const vec2 texCoord, 
	const int o, 
	const int w, 
	const int h, 
	const int storage
)
{
	const vec2 tc = vec2(float(( max(texCoord.x + 1000.0f, 0.0f) * w)) - 0.5f, float(( max(texCoord.y + 1000.0f, 0.0f) * h)) - 0.5f);
	const int iu = int(tc.x) % w;
	const int iv = int(tc.y) % h;
 #ifdef BILINEAR
	const float fu = tc.x - floor(tc.x);
	const float fv = tc.y - floor(tc.y);
	const float w0 = (1 - fu) * (1 - fv);
	const float w1 = fu * (1 - fv);
	const float w2 = (1 - fu) * fv;
	const float w3 = 1 - (w0 + w1 + w2);
	const uint iu1 = (iu + 1) % w;
	const uint iv1 = (iv + 1) %h;
	vec4 p0, p1, p2, p3;
	if (storage == ARGB32)
	{
		p0 = uintToVec4(ARGB32Data.pixels[o + iu + iv * w  ]);
		p1 = uintToVec4(ARGB32Data.pixels[o + iu1 + iv * w ]);
		p2 = uintToVec4(ARGB32Data.pixels[o + iu + iv1 * w ]);
		p3 = uintToVec4(ARGB32Data.pixels[o + iu1 + iv1 * w]);
	}
	else if (storage == ARGB128)
	{
		p0 = ARGB128Data.pixels[o + iu + iv * w  ];
		p1 = ARGB128Data.pixels[o + iu1 + iv * w ];
		p2 = ARGB128Data.pixels[o + iu + iv1 * w ];
		p3 = ARGB128Data.pixels[o + iu1 + iv1 * w];
	}
	else /* if (storage == NRM32) */
	{
		p0 = uintToVec4(NRM32Data.pixels[o + iu + iv * w  ]);
		p1 = uintToVec4(NRM32Data.pixels[o + iu1 + iv * w ]);
		p2 = uintToVec4(NRM32Data.pixels[o + iu + iv1 * w ]);
		p3 = uintToVec4(NRM32Data.pixels[o + iu1 + iv1 * w]);
	}
	return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
 #else
	if (storage == ARGB32) return uintToVec4( ARGB32Data.pixels[o + iu + iv * w] );
	else if (storage == ARGB128) return ARGB128Data.pixels[o + iu + iv * w];
	/* else if (storage == NRM32) */ return uintToVec4( NRM32Data.pixel[o + iu + iv * w] );
 #endif
}

vec4 FetchTexelTrilinear(const float lambda, const vec2 texCoord, const int offset, const int width, const int height)
{
	const int level0 = min(MIPLEVELCOUNT - 1, int(lambda));
	const int level1 = min(MIPLEVELCOUNT - 1, level0 + 1);
	const float f = lambda - floor(lambda);

	// Select first MIP level
	int o0 = offset, w0 = width, h0 = height;
	for (int i = 0; i < level0; i++) o0 += w0 * h0, w0 >>= 1, h0 >>= 1;
	// Select second MIP level
	int o1 = offset, w1 = width, h1 = height;
	for (int i = 0; i < level1; i++) o1 += w1 * h1, w1 >>= 1, h1 >>= 1;
	// Read actual data
	const vec4 p0 = FetchTexel(texCoord, o0, w0, h0, ARGB32);
	const vec4 p1 = FetchTexel(texCoord, o1, w1, h1, ARGB32);
	// Final interpolation
	return (1 - f) * p0 + f * p1;
}

#endif