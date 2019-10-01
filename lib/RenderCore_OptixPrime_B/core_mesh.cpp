/* core_mesh.cpp - Copyright 2019 Utrecht University

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

#include "core_settings.h"

RenderCore* CoreMesh::renderCore = 0;

static bool ConsistentExponents( const float4& ref, const float4& A, const float4& B )
{
	// retur true if the float3's stored in A and B have the same signs and exponents
	// (on a per-component basis) as the reference float3 stored in ref
	const uint& expRx = reinterpret_cast<const uint&>( ref.x ) >> 23;
	const uint& expRy = reinterpret_cast<const uint&>( ref.y ) >> 23;
	const uint& expRz = reinterpret_cast<const uint&>( ref.z ) >> 23;
	const uint& expAx = reinterpret_cast<const uint&>( A.x ) >> 23;
	const uint& expAy = reinterpret_cast<const uint&>( A.y ) >> 23;
	const uint& expAz = reinterpret_cast<const uint&>( A.z ) >> 23;
	const uint& expBx = reinterpret_cast<const uint&>( B.x ) >> 23;
	const uint& expBy = reinterpret_cast<const uint&>( B.y ) >> 23;
	const uint& expBz = reinterpret_cast<const uint&>( B.z ) >> 23;
	const bool same = (expAx == expRx) & (expAy == expRy) & (expAz == expRz) & (expBx == expRx) & (expBy == expRy) & (expBz == expRz);
	return same;
}

//  +-----------------------------------------------------------------------------+
//  |  CoreMesh::~CoreMesh                                                        |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
CoreMesh::~CoreMesh()
{
	delete triangles;
	rtpBufferDescDestroy( indicesDesc );
	rtpBufferDescDestroy( verticesDesc );
	rtpModelDestroy( model );
}

//  +-----------------------------------------------------------------------------+
//  |  CoreMesh::SetGeometry                                                      |
//  |  Set the geometry data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void CoreMesh::SetGeometry( const float4* vertexData, const int vertexCount, const int triCount, const CoreTri* tris, const uint* alphaFlags )
{
	// copy triangle data to GPU
	bool reallocate = (triangles == 0);
	if (triangles) if (triCount > triangles->GetSize()) reallocate = true;
	if (reallocate)
	{
		delete triangles;
		triangles = new CoreBuffer<CoreTri4>( triCount, ON_DEVICE, tris );
		// create dummy index data
		delete indexData;
		indexData = new uint3[triCount];
		for( int i = 0; i < triCount; i++ ) indexData[i] = make_uint3( i * 3 + 0, i * 3 + 1, i * 3 + 2 );
		// create float3 vertex data
		delete vertex3Data;
		vertex3Data = new float3[vertexCount];
		// create OptiX geometry buffers
		CHK_PRIME( rtpBufferDescCreate( RenderCore::context, RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_HOST, indexData, &indicesDesc ) );
		CHK_PRIME( rtpBufferDescCreate( RenderCore::context, RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, vertex3Data, &verticesDesc ) );
		CHK_PRIME( rtpBufferDescSetRange( indicesDesc, 0, triCount ) );
		CHK_PRIME( rtpBufferDescSetRange( verticesDesc, 0, vertexCount ) );
		// create model
		CHK_PRIME( rtpModelCreate( RenderCore::context, &model ) );
	}
	// copy new vertex positions and normals
	for( int i = 0; i < vertexCount; i++ ) vertex3Data[i] = make_float3( vertexData[i] );
	triangles->SetHostData( (CoreTri4*)tris );
	triangles->CopyToDevice();
	// update accstruc
	CHK_PRIME( rtpModelSetTriangles( model, indicesDesc, verticesDesc ) );
	CHK_PRIME( rtpModelUpdate( model, RTP_MODEL_HINT_NONE /* blocking; try RTP_MODEL_HINT_ASYNC + rtpModelFinish for async version. */ ) );
}

// EOF