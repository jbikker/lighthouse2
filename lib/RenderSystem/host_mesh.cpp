/* host_mesh.cpp - Copyright 2019 Utrecht University

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

#include "rendersystem.h"
#include "direct.h"

using namespace tinygltf;

static string GetFilePathExtension( string& fileName )
{
	if (fileName.find_last_of( "." ) != string::npos) return fileName.substr( fileName.find_last_of( "." ) + 1 );
	return "";
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::HostMesh                                                         |
//  |  Constructors.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostMesh::HostMesh( const char* file, const char* dir, const float scale )
{
	LoadGeometry( file, dir, scale );
}

HostMesh::HostMesh( tinygltfMesh& gltfMesh, tinygltfModel& gltfModel, const int matIdxOffset )
{
	ConvertFromGTLFMesh( gltfMesh, gltfModel, matIdxOffset );
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::~HostMesh                                                        |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
HostMesh::~HostMesh()
{
	// TODO: warn if instances using this mesh still exist?
	// And in general, do we want a two-way link between related objects?
	// - Materials and meshes;
	// - HostAreaLights and HostTris;
	// - Meshes and instances;
	// - Materials and textures;
	// - ...
	// Right now, a mesh has a list of materials, so we can efficiently remove
	// area lights when a material changes, or when an instance is removed. We
	// could do this for all related objects; in most cases this can be made
	// efficient.
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::LoadGeometry                                                     |
//  |  Load geometry data from disk.                                        LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMesh::LoadGeometry( const char* file, const char* dir, const float scale )
{
	// process supplied file name
	mat4 transform = mat4::Scale( scale ); // may include scale, translation, axis exchange
	string cleanFileName = LowerCase( dir + string( file ) ); // so we don't have to check for e.g. .OBJ
	string extension = GetFilePathExtension( cleanFileName );
	if (extension.compare( "obj" ) == 0)
	{
		LoadGeometryFromOBJ( cleanFileName.c_str(), dir, transform );
	}
	else
	{
		FatalError( __FILE__, __LINE__, "unsupported extension in file", cleanFileName.c_str() );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::LoadGeometryFromObj                                              |
//  |  Load an obj file using tinyobj.                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMesh::LoadGeometryFromOBJ( const string& fileName, const char* directory, const mat4& transform )
{
	// load obj file
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;
	map<string, GLuint> textures;
	string err;
	Timer timer;
	timer.reset();
	tinyobj::LoadObj( &attrib, &shapes, &materials, &err, fileName.c_str(), directory );
	if (err.size() > 0) FatalError( "problem loading %s:\n%s", fileName.c_str(), err.c_str() );
	printf( "loaded mesh in %5.3fs\n", timer.elapsed() );
	// material offset: if we loaded an object before this one, material indices should not start at 0.
	int matIdxOffset = (int)HostScene::materials.size();
	// process materials
	timer.reset();
	char currDir[1024];
	_getcwd( currDir, 1024 ); // GetCurrentDirectory( 1024, currDir );
	_chdir( directory ); // SetCurrentDirectory( directory );
	materialList.clear();
	materialList.reserve( materials.size() );
	for (auto &mtl : materials)
	{
		// initialize
		HostMaterial* material = new HostMaterial();
		material->ID = (int)HostScene::materials.size();
		material->origin = fileName;
		material->ConvertFrom( mtl );
		material->flags |= HostMaterial::FROM_MTL;
		HostScene::materials.push_back( material );
		materialList.push_back( material->ID );
	}
	_chdir( currDir ); // SetCurrentDirectory( currDir );
	printf( "materials finalized in %5.3fs\n", timer.elapsed() );
	// calculate values for consistent normal interpolation
	const uint verts = (uint)attrib.normals.size() / 3;
	vector<float> alphas;
	timer.reset();
	alphas.resize( verts, 1.0f ); // we will have one alpha value per unique vertex normal
	for (uint s = (uint)shapes.size(), i = 0; i < s; i++)
	{
		vector<tinyobj::index_t>& indices = shapes[i].mesh.indices;
		for (uint s = (uint)indices.size(), f = 0; f < s; f += 3)
		{
			const int idx0 = indices[f + 0].vertex_index, nidx0 = indices[f + 0].normal_index;
			const int idx1 = indices[f + 1].vertex_index, nidx1 = indices[f + 1].normal_index;
			const int idx2 = indices[f + 2].vertex_index, nidx2 = indices[f + 2].normal_index;
			const float3 vert0 = make_float3( attrib.vertices[idx0 * 3 + 0], attrib.vertices[idx0 * 3 + 1], attrib.vertices[idx0 * 3 + 2] );
			const float3 vert1 = make_float3( attrib.vertices[idx1 * 3 + 0], attrib.vertices[idx1 * 3 + 1], attrib.vertices[idx1 * 3 + 2] );
			const float3 vert2 = make_float3( attrib.vertices[idx2 * 3 + 0], attrib.vertices[idx2 * 3 + 1], attrib.vertices[idx2 * 3 + 2] );
			const float3 vN0 = make_float3( attrib.normals[nidx0 * 3 + 0], attrib.normals[nidx0 * 3 + 1], attrib.normals[nidx0 * 3 + 2] );
			const float3 vN1 = make_float3( attrib.normals[nidx1 * 3 + 0], attrib.normals[nidx1 * 3 + 1], attrib.normals[nidx1 * 3 + 2] );
			const float3 vN2 = make_float3( attrib.normals[nidx2 * 3 + 0], attrib.normals[nidx2 * 3 + 1], attrib.normals[nidx2 * 3 + 2] );
			float3 N = normalize( cross( vert1 - vert0, vert2 - vert0 ) );
			if (dot( N, vN0 ) < 0 && dot( N, vN1 ) < 0 && dot( N, vN2 ) < 0) N *= -1.0f; // flip if not consistent with vertex normals
			// loop over vertices
			// Note: we clamp at approx. 45 degree angles; beyond this the approach fails.
			alphas[nidx0] = min( alphas[nidx0], max( 0.7f, dot( vN0, N ) ) );
			alphas[nidx1] = min( alphas[nidx1], max( 0.7f, dot( vN1, N ) ) );
			alphas[nidx2] = min( alphas[nidx2], max( 0.7f, dot( vN2, N ) ) );
		}
	}
	// finalize alpha values based on max dots
	const float w = 0.03632f;
	for (uint i = 0; i < verts; i++)
	{
		const float nnv = alphas[i]; // temporarily stored there
		alphas[i] = acosf( nnv ) * (1 + w * (1 - nnv) * (1 - nnv));
	}
	printf( "calculated vertex alphas in %5.3fs\n", timer.elapsed() );
	// extract data for ray tracing: raw vertex and index data
	aabb sceneBounds;
	int toReserve = 0;
	timer.reset();
	for (int s = (int)shapes.size(), idx = 0, i = 0; i < s; i++) toReserve += (int)shapes[i].mesh.indices.size();
	vertices.reserve( toReserve );
	indices.reserve( toReserve / 3 );
	for (int s = (int)shapes.size(), idx = 0, i = 0; i < s; i++) for (int f = 0; f < shapes[i].mesh.indices.size(); f += 3, idx += 3)
	{
		const uint idx0 = shapes[i].mesh.indices[f + 0].vertex_index;
		const uint idx1 = shapes[i].mesh.indices[f + 1].vertex_index;
		const uint idx2 = shapes[i].mesh.indices[f + 2].vertex_index;
		const float3 v0 = make_float3( attrib.vertices[idx0 * 3 + 0], attrib.vertices[idx0 * 3 + 1], attrib.vertices[idx0 * 3 + 2] );
		const float3 v1 = make_float3( attrib.vertices[idx1 * 3 + 0], attrib.vertices[idx1 * 3 + 1], attrib.vertices[idx1 * 3 + 2] );
		const float3 v2 = make_float3( attrib.vertices[idx2 * 3 + 0], attrib.vertices[idx2 * 3 + 1], attrib.vertices[idx2 * 3 + 2] );
		indices.push_back( make_uint3( idx, idx + 1, idx + 2 ) );
		const float4 tv0 = make_float4( v0, 1 ) * transform;
		const float4 tv1 = make_float4( v1, 1 ) * transform;
		const float4 tv2 = make_float4( v2, 1 ) * transform;
		vertices.push_back( tv0 );
		vertices.push_back( tv1 );
		vertices.push_back( tv2 );
		sceneBounds.Grow( make_float3( tv0 ) );
		sceneBounds.Grow( make_float3( tv1 ) );
		sceneBounds.Grow( make_float3( tv2 ) );
	}
	printf( "created polygon soup in %5.3fs\n", timer.elapsed() );
	printf( "scene bounds: (%5.2f,%5.2f,%5.2f)-(%5.2f,%5.2f,%5.2f)\n",
		sceneBounds.bmin3.x, sceneBounds.bmin3.y, sceneBounds.bmin3.z,
		sceneBounds.bmax3.x, sceneBounds.bmax3.y, sceneBounds.bmax3.z );
	// extract full model data and materials
	timer.reset();
	triangles.resize( indices.size() );
	for (int s = (int)shapes.size(), face = 0, i = 0; i < s; i++)
	{
		vector<tinyobj::index_t>& indices = shapes[i].mesh.indices;
		for (int s = (int)shapes[i].mesh.indices.size(), f = 0; f < s; f += 3, face++)
		{
			HostTri& tri = triangles[face];
			tri.vertex0 = make_float3( vertices[face * 3 + 0] );
			tri.vertex1 = make_float3( vertices[face * 3 + 1] );
			tri.vertex2 = make_float3( vertices[face * 3 + 2] );
			const int tidx0 = indices[f + 0].texcoord_index, nidx0 = indices[f + 0].normal_index, idx0 = indices[f + 0].vertex_index;
			const int tidx1 = indices[f + 1].texcoord_index, nidx1 = indices[f + 1].normal_index, idx1 = indices[f + 1].vertex_index;
			const int tidx2 = indices[f + 2].texcoord_index, nidx2 = indices[f + 2].normal_index, idx2 = indices[f + 2].vertex_index;
			tri.vN0 = make_float3( attrib.normals[nidx0 * 3 + 0], attrib.normals[nidx0 * 3 + 1], attrib.normals[nidx0 * 3 + 2] );
			tri.vN1 = make_float3( attrib.normals[nidx1 * 3 + 0], attrib.normals[nidx1 * 3 + 1], attrib.normals[nidx1 * 3 + 2] );
			tri.vN2 = make_float3( attrib.normals[nidx2 * 3 + 0], attrib.normals[nidx2 * 3 + 1], attrib.normals[nidx2 * 3 + 2] );
			const float3 e1 = tri.vertex1 - tri.vertex0;
			const float3 e2 = tri.vertex2 - tri.vertex0;
			float3 N = normalize( cross( e1, e2 ) );
			if (dot( N, tri.vN0 ) < 0) N *= -1.0f; // flip face normal if not consistent with vertex normal
			if (tidx0 > -1)
			{
				tri.u0 = attrib.texcoords[tidx0 * 2 + 0], tri.v0 = attrib.texcoords[tidx0 * 2 + 1];
				tri.u1 = attrib.texcoords[tidx1 * 2 + 0], tri.v1 = attrib.texcoords[tidx1 * 2 + 1];
				tri.u2 = attrib.texcoords[tidx2 * 2 + 0], tri.v2 = attrib.texcoords[tidx2 * 2 + 1];
				// calculate tangent vectors
				float2 uv01 = make_float2( tri.u1 - tri.u0, tri.v1 - tri.v0 );
				float2 uv02 = make_float2( tri.u2 - tri.u0, tri.v2 - tri.v0 );
				if (dot( uv01, uv01 ) == 0 || dot( uv02, uv02 ) == 0)
				{
					tri.T = normalize( tri.vertex1 - tri.vertex0 );
					tri.B = normalize( cross( N, tri.T ) );
				}
				else
				{
					tri.T = normalize( e1 * uv02.y - e2 * uv01.y );
					tri.B = normalize( e2 * uv01.x - e1 * uv02.x );
				}
			}
			else
			{
				tri.T = normalize( e1 );
				tri.B = normalize( cross( N, tri.T ) );
			}
			tri.Nx = N.x, tri.Ny = N.y, tri.Nz = N.z;
			tri.material = shapes[i].mesh.material_ids[f / 3] + matIdxOffset;
		#if 0
			const float a = (tri.vertex1 - tri.vertex0).length();
			const float b = (tri.vertex2 - tri.vertex1).length();
			const float c = (tri.vertex0 - tri.vertex2).length();
			const float s = (a + b + c) * 0.5f;
			tri.area = sqrtf( s * (s - a) * (s - b) * (s - c) ); // Heron's formula
		#else
			tri.area = 0; // we don't actually use it, except for lights, where it is also calculated
		#endif
			tri.invArea = 0; // todo
			tri.alpha = make_float3( alphas[nidx0], tri.alpha.y = alphas[nidx1], tri.alpha.z = alphas[nidx2] );
			// calculate triangle LOD data
			HostMaterial* mat = HostScene::materials[tri.material];
			int textureID = mat->map[TEXTURE0].textureID;
			if (textureID > -1)
			{
				HostTexture* texture = HostScene::textures[textureID];
				float Ta = (float)(texture->width * texture->height) * fabs( (tri.u1 - tri.u0) * (tri.v2 - tri.v0) - (tri.u2 - tri.u0) * (tri.v1 - tri.v0) );
				float Pa = length( cross( tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0 ) );
				tri.LOD = 0.5f * log2f( Ta / Pa );
			}
		}
	}
	printf( "verbose triangle data in %5.3fs\n", timer.elapsed() );
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::ConvertFromGTLFMesh                                              |
//  |  Convert a gltf mesh to a HostMesh.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMesh::ConvertFromGTLFMesh( tinygltfMesh& gltfMesh, tinygltfModel& gltfModel, const int matIdxOffset )
{
	const int targetCount = (int)gltfMesh.weights.size();
	for (int s = (int)gltfMesh.primitives.size(), j = 0; j < s; j++)
	{
		Primitive& prim = gltfMesh.primitives[j];
		size_t vertIdxOffset = vertices.size();
		size_t indexOffset = indices.size();
		// load indices
		const Accessor& indicesAccessor = gltfModel.accessors[prim.indices];
		const BufferView& bufferView = gltfModel.bufferViews[indicesAccessor.bufferView];
		const Buffer& buffer = gltfModel.buffers[bufferView.buffer];
		const uchar* a /* brevity */ = buffer.data.data() + bufferView.byteOffset + indicesAccessor.byteOffset;
		const int byteStride = indicesAccessor.ByteStride( bufferView );
		const size_t count = indicesAccessor.count;
		// allocate the index array in the pointer-to-base declared in the parent scope
		vector<int> tmpIndices;
		vector<float3> normals, tmpVertices;
		vector<float2> uvs;
		switch (indicesAccessor.componentType)
		{
		case TINYGLTF_COMPONENT_TYPE_BYTE: for (int k = 0; k < count; k++, a += byteStride) tmpIndices.push_back( *((char*)a) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: for (int k = 0; k < count; k++, a += byteStride) tmpIndices.push_back( *((uchar*)a) ); break;
		case TINYGLTF_COMPONENT_TYPE_SHORT: for (int k = 0; k < count; k++, a += byteStride) tmpIndices.push_back( *((short*)a) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: for (int k = 0; k < count; k++, a += byteStride) tmpIndices.push_back( *((ushort*)a) ); break;
		case TINYGLTF_COMPONENT_TYPE_INT: for (int k = 0; k < count; k++, a += byteStride) tmpIndices.push_back( *((int*)a) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: for (int k = 0; k < count; k++, a += byteStride) tmpIndices.push_back( *((uint*)a) ); break;
		default: break;
		}
		// turn into faces - re-arrange the indices so that it describes a simple list of triangles
		if (prim.mode == TINYGLTF_MODE_TRIANGLE_FAN)
		{
			vector<int> fan = move( tmpIndices );
			tmpIndices.clear();
			for (size_t s = fan.size(), i = 2; i < s; i++)
			{
				tmpIndices.push_back( fan[0] );
				tmpIndices.push_back( fan[i - 1] );
				tmpIndices.push_back( fan[i] );
			}
		}
		else if (prim.mode == TINYGLTF_MODE_TRIANGLE_STRIP)
		{
			vector<int> strip = move( tmpIndices );
			tmpIndices.clear();
			for (size_t s = strip.size(), i = 2; i < s; i++)
			{
				tmpIndices.push_back( strip[i - 2] );
				tmpIndices.push_back( strip[i - 1] );
				tmpIndices.push_back( strip[i] );
			}
		}
		else if (prim.mode != TINYGLTF_MODE_TRIANGLES) /* skipping non-triangle primitive. */ continue;
		// we now have a simple list of vertex indices, 3 per triangle (TINYGLTF_MODE_TRIANGLES)
		for (const auto& attribute : prim.attributes)
		{
			const Accessor attribAccessor = gltfModel.accessors[attribute.second];
			const BufferView& bufferView = gltfModel.bufferViews[attribAccessor.bufferView];
			const Buffer& buffer = gltfModel.buffers[bufferView.buffer];
			const uchar* a = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
			const int byte_stride = attribAccessor.ByteStride( bufferView );
			const size_t count = attribAccessor.count;
			if (attribute.first == "POSITION")
			{
				float3 boundsMin = make_float3( attribAccessor.minValues[0], attribAccessor.minValues[1], attribAccessor.minValues[2] );
				float3 boundsMax = make_float3( attribAccessor.maxValues[0], attribAccessor.maxValues[1], attribAccessor.maxValues[2] );
				if (attribAccessor.type == TINYGLTF_TYPE_VEC3)
					if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						for (size_t i = 0; i < count; i++, a += byte_stride) tmpVertices.push_back( *((float3*)a) );
					else FatalError( __FILE__, __LINE__, "double precision positions not supported in gltf file", "" );
				else FatalError( __FILE__, __LINE__, "unsupported position definition in gltf file", "" );
			}
			if (attribute.first == "NORMAL")
			{
				if (attribAccessor.type == TINYGLTF_TYPE_VEC3)
					if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						for (size_t i = 0; i < count; i++, a += byte_stride) normals.push_back( *((float3*)a) );
					else FatalError( __FILE__, __LINE__, "double precision normals not supported in gltf file", "" );
				else FatalError( __FILE__, __LINE__, "expected vec3 normals in gltf file", "" );
			}
			if (attribute.first == "TANGENT") /* not yet supported */ continue;
			if (attribute.first == "TEXCOORD_0")
			{
				if (attribAccessor.type == TINYGLTF_TYPE_VEC2)
					if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						for (size_t i = 0; i < count; i++, a += byte_stride) uvs.push_back( *((float2*)a) );
					else FatalError( __FILE__, __LINE__, "double precision uvs not supported in gltf file", "" );
				else FatalError( __FILE__, __LINE__, "expected vec2 uvs in gltf file", "" );
			}
		}
		// obtain morph targets
		if (targetCount > 0)
		{
			// store base pose
			poses.push_back( new Pose() );
			for (int s = (int)tmpVertices.size(), i = 0; i < s; i++)
			{
				poses[0]->positions.push_back( tmpVertices[i] );
				poses[0]->normals.push_back( normals[i] );
				poses[0]->tangents.push_back( make_float3( 0 ) /* TODO */ );
			}
		}
		for (int i = 0; i < targetCount; i++)
		{
			// https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#morph-targets
			poses.push_back( new Pose() );
			for (const auto& target : prim.targets[i])
			{
				const Accessor targetAccessor = gltfModel.accessors[target.second];
				const BufferView& bufferView = gltfModel.bufferViews[targetAccessor.bufferView];
				const Buffer& buffer = gltfModel.buffers[bufferView.buffer];
				const uchar* a = buffer.data.data() + bufferView.byteOffset + targetAccessor.byteOffset;
				const int stride = targetAccessor.ByteStride( bufferView );
				assert( targetAccessor.count == tmpVertices.size() );
				assert( targetAccessor.type == TINYGLTF_TYPE_VEC3 );
				assert( targetAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT );
				for (int j = 0; j < targetAccessor.count; j++, a += 3 * stride)
				{
					float3 v = make_float3( *((float*)a), *((float*)(a + stride)), *((float*)(a + 2 * stride)) );
					if (target.first == "POSITION") poses[i + 1]->positions.push_back( v );
					if (target.first == "NORMAL") poses[i + 1]->normals.push_back( v );
					if (target.first == "TANGENT") poses[i + 1]->tangents.push_back( v );
				}
			}
		}
		// calculate values for consistent normal interpolation
		vector<float> alphas;
		alphas.resize( normals.size(), 1.0f ); // we will have one alpha value per unique vertex normal
		for (size_t s = tmpIndices.size(), i = 0; i < s; i += 3)
		{
			const uint v0idx = tmpIndices[i + 0], v1idx = tmpIndices[i + 1], v2idx = tmpIndices[i + 2];
			const float3 vert0 = tmpVertices[v0idx], vert1 = tmpVertices[v1idx], vert2 = tmpVertices[v2idx];
			const float3 vN0 = normals[v0idx], vN1 = normals[v1idx], vN2 = normals[v2idx];
			float3 N = normalize( cross( vert1 - vert0, vert2 - vert0 ) );
			if (dot( N, vN0 ) < 0 && dot( N, vN1 ) < 0 && dot( N, vN2 ) < 0) N *= -1.0f; // flip if not consistent with vertex normals
			// Note: we clamp at approx. 45 degree angles; beyond this the approach fails.
			alphas[v0idx] = min( alphas[v0idx], max( 0.7f, dot( vN0, N ) ) );
			alphas[v1idx] = min( alphas[v0idx], max( 0.7f, dot( vN1, N ) ) );
			alphas[v2idx] = min( alphas[v0idx], max( 0.7f, dot( vN2, N ) ) );
		}
		for (size_t s = alphas.size(), i = 0; i < s; i++)
		{
			const float nnv = alphas[i]; // temporarily stored there
			alphas[i] = acosf( nnv ) * (1 + 0.03632f * (1 - nnv) * (1 - nnv));
		}
		// all data has been read; add triangles to the HostMesh
		const size_t newTriangleCount = tmpIndices.size() / 3;
		size_t triIdx = triangles.size();
		triangles.resize( triIdx + newTriangleCount );
		for (size_t i = 0; i < newTriangleCount; i++, triIdx++)
		{
			HostTri& tri = triangles[triIdx];
			const uint v0idx = tmpIndices[i * 3 + 0];
			const uint v1idx = tmpIndices[i * 3 + 1];
			const uint v2idx = tmpIndices[i * 3 + 2];
			const int vertIdx = (int)vertices.size();
			indices.push_back( make_uint3( vertIdx, vertIdx + 1, vertIdx + 2 ) );
			const float3 v0pos = tmpVertices[v0idx];
			const float3 v1pos = tmpVertices[v1idx];
			const float3 v2pos = tmpVertices[v2idx];
			vertices.push_back( make_float4( v0pos, 1 ) );
			vertices.push_back( make_float4( v1pos, 1 ) );
			vertices.push_back( make_float4( v2pos, 1 ) );
			const float3 N = normalize( cross( v1pos - v0pos, v2pos - v0pos ) );
			tri.Nx = N.x, tri.Ny = N.y, tri.Nz = N.z;
			tri.vertex0 = tmpVertices[v0idx];
			tri.vertex1 = tmpVertices[v1idx];
			tri.vertex2 = tmpVertices[v2idx];
			tri.alpha = make_float3( alphas[v0idx], alphas[v1idx], alphas[v2idx] );
			if (normals.size() > 0)
				tri.vN0 = normals[v0idx],
				tri.vN1 = normals[v1idx],
				tri.vN2 = normals[v2idx];
			if (uvs.size() > 0)
			{
				tri.u0 = uvs[v0idx].x, tri.v0 = uvs[v0idx].y;
				tri.u1 = uvs[v1idx].x, tri.v1 = uvs[v1idx].y;
				tri.u2 = uvs[v2idx].x, tri.v2 = uvs[v2idx].y;
				// calculate tangent vector based on uvs
				float2 uv01 = make_float2( tri.u1 - tri.u0, tri.v1 - tri.v0 );
				float2 uv02 = make_float2( tri.u2 - tri.u0, tri.v2 - tri.v0 );
				if (dot( uv01, uv01 ) == 0 || dot( uv02, uv02 ) == 0)
				{
					tri.T = normalize( tri.vertex1 - tri.vertex0 );
					tri.B = normalize( cross( N, tri.T ) );
				}
				else
				{
					// uvs cannot be used; use edges instead
					tri.T = normalize( (tri.vertex1 - tri.vertex0) * uv02.y - (tri.vertex2 - tri.vertex0) * uv01.y );
					tri.B = normalize( (tri.vertex2 - tri.vertex0) * uv01.x - (tri.vertex1 - tri.vertex0) * uv02.x );
				}
			}
			else
			{
				// no uv information; use edges to calculate tangent vectors
				tri.T = normalize( tri.vertex1 - tri.vertex0 );
				tri.B = normalize( cross( N, tri.T ) );
			}
			tri.material = prim.material + matIdxOffset;
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::BuildMaterialList                                                |
//  |  Update the list of materials used by this mesh. We will use this list to   |
//  |  efficiently find meshes using a specific material, which in turn is useful |
//  |  when a material becomes emissive or non-emissive.                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMesh::BuildMaterialList()
{
	// mark all materials as 'not seen yet'
	for (auto material : HostScene::materials) material->visited = false;
	// add each material
	materialList.clear();
	for (auto tri : triangles)
	{
		HostMaterial* material = HostScene::materials[tri.material];
		if (!material->visited)
		{
			material->visited = true;
			materialList.push_back( material->ID );
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::UpdateAlphaFlags                                                 |
//  |  Create or update the list of alpha flags; one is set to true or fale for   |
//  |  each triangle in the mesh. This will later be used to mark triangles in    |
//  |  the core in a core-specific way, and ultimately, to detect triangles that  |
//  |  may have alpha transparency as efficiently as possible during              |
//  |  traversal.                                                           LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMesh::UpdateAlphaFlags()
{
	const uint triCount = (uint)triangles.size();
	if (alphaFlags.size() != triCount) alphaFlags.resize( triCount, 0 );
	for (uint i = 0; i < triCount; i++)
		if (HostScene::materials[triangles[i].material]->flags & HostMaterial::HASALPHA)
			alphaFlags[i] = 1;
}

// EOF