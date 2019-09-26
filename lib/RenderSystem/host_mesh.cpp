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
//  |  HostSkin::HostSkin                                                         |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostSkin::HostSkin( const tinygltfSkin& gltfSkin, const tinygltfModel& gltfModel, const int nodeBase )
{
	ConvertFromGLTFSkin( gltfSkin, gltfModel, nodeBase );
}

//  +-----------------------------------------------------------------------------+
//  |  HostSkin::HostSkin                                                         |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
void HostSkin::ConvertFromGLTFSkin( const tinygltfSkin& gltfSkin, const tinygltfModel& gltfModel, const int nodeBase )
{
	name = gltfSkin.name;
	skeletonRoot = (gltfSkin.skeleton == -1 ? 0 : gltfSkin.skeleton) + nodeBase;
	for (int jointIndex : gltfSkin.joints) joints.push_back( jointIndex + nodeBase );
	if (gltfSkin.inverseBindMatrices > -1)
	{
		const auto& accessor = gltfModel.accessors[gltfSkin.inverseBindMatrices];
		const auto& bufferView = gltfModel.bufferViews[accessor.bufferView];
		const auto& buffer = gltfModel.buffers[bufferView.buffer];
		inverseBindMatrices.resize( accessor.count );
		memcpy( inverseBindMatrices.data(), &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof( mat4 ) );
		jointMat.resize( accessor.count );
		// convert gltf's column-major to row-major
		for( int k = 0; k < accessor.count; k++ ) 
		{
			mat4 M = inverseBindMatrices[k];
			for( int i = 0; i < 4; i++ ) for( int j = 0; j < 4; j++ ) inverseBindMatrices[k].cell[j * 4 + i] = M.cell[i * 4 + j];
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::HostMesh                                                         |
//  |  Constructors.                                                        LH2'19|
//  +-----------------------------------------------------------------------------+
HostMesh::HostMesh( const char* file, const char* dir, const float scale )
{
	LoadGeometry( file, dir, scale );
}

HostMesh::HostMesh( const tinygltfMesh& gltfMesh, const tinygltfModel& gltfModel, const int matIdxOffset, const int materialOverride )
{
	ConvertFromGTLFMesh( gltfMesh, gltfModel, matIdxOffset, materialOverride );
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
	if (err.size() > 0) FatalError( __FILE__, __LINE__, err.c_str(), fileName.c_str() );
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
	for (int s = (int)shapes.size(), i = 0; i < s; i++) toReserve += (int)shapes[i].mesh.indices.size();
	vertices.reserve( toReserve );
	for (int s = (int)shapes.size(), i = 0; i < s; i++) for (int f = 0; f < shapes[i].mesh.indices.size(); f += 3)
	{
		const uint idx0 = shapes[i].mesh.indices[f + 0].vertex_index;
		const uint idx1 = shapes[i].mesh.indices[f + 1].vertex_index;
		const uint idx2 = shapes[i].mesh.indices[f + 2].vertex_index;
		const float3 v0 = make_float3( attrib.vertices[idx0 * 3 + 0], attrib.vertices[idx0 * 3 + 1], attrib.vertices[idx0 * 3 + 2] );
		const float3 v1 = make_float3( attrib.vertices[idx1 * 3 + 0], attrib.vertices[idx1 * 3 + 1], attrib.vertices[idx1 * 3 + 2] );
		const float3 v2 = make_float3( attrib.vertices[idx2 * 3 + 0], attrib.vertices[idx2 * 3 + 1], attrib.vertices[idx2 * 3 + 2] );
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
	triangles.resize( vertices.size() / 3 );
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
void HostMesh::ConvertFromGTLFMesh( const tinygltfMesh& gltfMesh, const tinygltfModel& gltfModel, const int matIdxOffset, const int materialOverride )
{
	const int targetCount = (int)gltfMesh.weights.size();
	for (int s = (int)gltfMesh.primitives.size(), j = 0; j < s; j++)
	{
		const Primitive& prim = gltfMesh.primitives[j];
		// load indices
		const Accessor& accessor = gltfModel.accessors[prim.indices];
		const BufferView& view = gltfModel.bufferViews[accessor.bufferView];
		const Buffer& buffer = gltfModel.buffers[view.buffer];
		const uchar* a /* brevity */ = buffer.data.data() + view.byteOffset + accessor.byteOffset;
		const int byteStride = accessor.ByteStride( view );
		const size_t count = accessor.count;
		// allocate the index array in the pointer-to-base declared in the parent scope
		vector<int> tmpIndices;
		vector<float3> tmpNormals, tmpVertices;
		vector<float2> tmpUvs;
		vector<uint4> tmpJoints;
		vector<float4> tmpWeights;
		switch (accessor.componentType)
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
					else FatalError( __FILE__, __LINE__, "double precision positions not supported in gltf file" );
				else FatalError( __FILE__, __LINE__, "unsupported position definition in gltf file" );
			}
			else if (attribute.first == "NORMAL")
			{
				if (attribAccessor.type == TINYGLTF_TYPE_VEC3)
					if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						for (size_t i = 0; i < count; i++, a += byte_stride) tmpNormals.push_back( *((float3*)a) );
					else FatalError( __FILE__, __LINE__, "double precision normals not supported in gltf file", "" );
				else FatalError( __FILE__, __LINE__, "expected vec3 normals in gltf file", "" );
			}
			else if (attribute.first == "TANGENT") /* not yet supported */ continue;
			else if (attribute.first == "TEXCOORD_0")
			{
				if (attribAccessor.type == TINYGLTF_TYPE_VEC2)
					if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						for (size_t i = 0; i < count; i++, a += byte_stride) tmpUvs.push_back( *((float2*)a) );
					else FatalError( __FILE__, __LINE__, "double precision uvs not supported in gltf file", "" );
				else FatalError( __FILE__, __LINE__, "expected vec2 uvs in gltf file", "" );
			}
			else if (attribute.first == "TEXCOORD_1")
			{
				// TODO; ignored for now.
			}
			else if (attribute.first == "COLOR_0")
			{
				// TODO; ignored for now.
			}
			else if (attribute.first == "JOINTS_0")
			{
				if (attribAccessor.type == TINYGLTF_TYPE_VEC4)
					if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
						for (size_t i = 0; i < count; i++, a += byte_stride) 
							tmpJoints.push_back( make_uint4( *((ushort*)a), *((ushort*)(a + 2)), *((ushort*)(a + 4)), *((ushort*)(a + 6)) ) );
					else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
						for (size_t i = 0; i < count; i++, a += byte_stride) 
							tmpJoints.push_back( make_uint4( *((uchar*)a), *((uchar*)(a + 1)), *((uchar*)(a + 2)), *((uchar*)(a + 3)) ) );
					else FatalError( __FILE__, __LINE__, "expected ushorts or uchars for joints in gltf file", "" );
				else FatalError( __FILE__, __LINE__, "expected vec4s for joints in gltf file", "" );
			}
			else if (attribute.first == "WEIGHTS_0")
			{
				if (attribAccessor.type == TINYGLTF_TYPE_VEC4)
					if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						for (size_t i = 0; i < count; i++, a += byte_stride) 
						{
							float4 w4 = *((float4*)a);
							float norm = 1.0f / (w4.x + w4.y + w4.z + w4.w);
							w4 *= norm;
							tmpWeights.push_back( w4 );
						}
					else FatalError( __FILE__, __LINE__, "double precision uvs not supported in gltf file", "" );
				else FatalError( __FILE__, __LINE__, "expected vec4 weights in gltf file", "" );
			}
			else assert( false ); // unkown property
		}
		// obtain morph targets
		vector<Pose> tmpPoses;
		if (targetCount > 0)
		{
			// store base pose
			tmpPoses.push_back( Pose() );
			for (int s = (int)tmpVertices.size(), i = 0; i < s; i++)
			{
				tmpPoses[0].positions.push_back( tmpVertices[i] );
				tmpPoses[0].normals.push_back( tmpNormals[i] );
				tmpPoses[0].tangents.push_back( make_float3( 0 ) /* TODO */ );
			}
		}
		for (int i = 0; i < targetCount; i++)
		{
			tmpPoses.push_back( Pose() );
			for (const auto& target : prim.targets[i])
			{
				const Accessor accessor = gltfModel.accessors[target.second];
				const BufferView& view = gltfModel.bufferViews[accessor.bufferView];
				const float* a = (const float*)(gltfModel.buffers[view.buffer].data.data() + view.byteOffset + accessor.byteOffset);
				for (int j = 0; j < accessor.count; j++)
				{
					float3 v = make_float3( a[j * 3], a[j * 3 + 1], a[j * 3 + 2] );
					if (target.first == "POSITION") tmpPoses[i + 1].positions.push_back( v );
					if (target.first == "NORMAL") tmpPoses[i + 1].normals.push_back( v );
					if (target.first == "TANGENT") tmpPoses[i + 1].tangents.push_back( v );
				}
			}
		}
		// all data has been read; add triangles to the HostMesh
		BuildFromIndexedData( tmpIndices, tmpVertices, tmpNormals, tmpUvs, tmpPoses,
			tmpJoints, tmpWeights, materialOverride == -1 ? (prim.material + matIdxOffset) : materialOverride );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::BuildFromIndexedData                                             |
//  |  We use non-indexed triangles, so three subsequent vertices form a tri,     |
//  |  to skip one indirection during intersection. glTF and obj store indexed    |
//  |  data, which we now convert to the final representation.              LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMesh::BuildFromIndexedData( const vector<int>& tmpIndices, const vector<float3>& tmpVertices,
	const vector<float3>& tmpNormals, const vector<float2>& tmpUvs, const vector<Pose>& tmpPoses,
	const vector<uint4>& tmpJoints, const vector<float4>& tmpWeights, const int materialIdx )
{
	// calculate values for consistent normal interpolation
	vector<float> tmpAlphas;
	tmpAlphas.resize( tmpVertices.size(), 1.0f ); // we will have one alpha value per unique vertex
	for (size_t s = tmpIndices.size(), i = 0; i < s; i += 3)
	{
		const uint v0idx = tmpIndices[i + 0], v1idx = tmpIndices[i + 1], v2idx = tmpIndices[i + 2];
		const float3 vert0 = tmpVertices[v0idx], vert1 = tmpVertices[v1idx], vert2 = tmpVertices[v2idx];
		float3 N = normalize( cross( vert1 - vert0, vert2 - vert0 ) );
		float3 vN0, vN1, vN2;
		if (tmpNormals.size() > 0) 
		{
			vN0 = tmpNormals[v0idx], vN1 = tmpNormals[v1idx], vN2 = tmpNormals[v2idx]; 
			if (dot( N, vN0 ) < 0 && dot( N, vN1 ) < 0 && dot( N, vN2 ) < 0) N *= -1.0f; // flip if not consistent with vertex normals
		}
		else
		{
			// no normals supplied; copy face normal
			vN0 = vN1 = vN2 = N;
		}
		// Note: we clamp at approx. 45 degree angles; beyond this the approach fails.
		tmpAlphas[v0idx] = min( tmpAlphas[v0idx], max( 0.7f, dot( vN0, N ) ) );
		tmpAlphas[v1idx] = min( tmpAlphas[v0idx], max( 0.7f, dot( vN1, N ) ) );
		tmpAlphas[v2idx] = min( tmpAlphas[v0idx], max( 0.7f, dot( vN2, N ) ) );
	}
	for (size_t s = tmpAlphas.size(), i = 0; i < s; i++)
	{
		const float nnv = tmpAlphas[i]; // temporarily stored there
		tmpAlphas[i] = acosf( nnv ) * (1 + 0.03632f * (1 - nnv) * (1 - nnv));
	}
	// build final mesh structures
	const size_t newTriangleCount = tmpIndices.size() / 3;
	size_t triIdx = triangles.size();
	triangles.resize( triIdx + newTriangleCount );
	for (size_t i = 0; i < newTriangleCount; i++, triIdx++)
	{
		HostTri& tri = triangles[triIdx];
		const uint v0idx = tmpIndices[i * 3 + 0];
		const uint v1idx = tmpIndices[i * 3 + 1];
		const uint v2idx = tmpIndices[i * 3 + 2];
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
		tri.alpha = make_float3( tmpAlphas[v0idx], tmpAlphas[v1idx], tmpAlphas[v2idx] );
		if (tmpNormals.size() > 0)
			tri.vN0 = tmpNormals[v0idx],
			tri.vN1 = tmpNormals[v1idx],
			tri.vN2 = tmpNormals[v2idx];
		else
			tri.vN0 = tri.vN1 = tri.vN2 = N;
		if (tmpUvs.size() > 0)
		{
			tri.u0 = tmpUvs[v0idx].x, tri.v0 = tmpUvs[v0idx].y;
			tri.u1 = tmpUvs[v1idx].x, tri.v1 = tmpUvs[v1idx].y;
			tri.u2 = tmpUvs[v2idx].x, tri.v2 = tmpUvs[v2idx].y;
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
		tri.material = materialIdx;
	}
	// build poses
	if (tmpPoses.size() > 0)
	{
		for (int s = (int)tmpPoses.size(), i = 0; i < s; i++)
		{
			poses.push_back( Pose() );
			for (int s = (int)triangles.size(), j = 0; j < s; j++)
			{
				const uint v0idx = tmpIndices[j * 3 + 0], v1idx = tmpIndices[j * 3 + 1], v2idx = tmpIndices[j * 3 + 2];
				poses[i].positions.push_back( tmpPoses[i].positions[v0idx] );
				poses[i].positions.push_back( tmpPoses[i].positions[v1idx] );
				poses[i].positions.push_back( tmpPoses[i].positions[v2idx] );
				poses[i].normals.push_back( tmpPoses[i].normals[v0idx] );
				poses[i].normals.push_back( tmpPoses[i].normals[v1idx] );
				poses[i].normals.push_back( tmpPoses[i].normals[v2idx] );
				poses[i].tangents.push_back( tmpPoses[i].tangents[v0idx] );
				poses[i].tangents.push_back( tmpPoses[i].tangents[v1idx] );
				poses[i].tangents.push_back( tmpPoses[i].tangents[v2idx] );
			}
		}
	}
	// process joints / weights
	if (tmpJoints.size() > 0) for (int s = (int)triangles.size(), j = 0; j < s; j++)
	{
		const uint v0idx = tmpIndices[j * 3 + 0], v1idx = tmpIndices[j * 3 + 1], v2idx = tmpIndices[j * 3 + 2];
		joints.push_back( tmpJoints[v0idx] );
		joints.push_back( tmpJoints[v1idx] );
		joints.push_back( tmpJoints[v2idx] );
		weights.push_back( tmpWeights[v0idx] );
		weights.push_back( tmpWeights[v1idx] );
		weights.push_back( tmpWeights[v2idx] );
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

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::SetPose                                                          |
//  |  Update the geometry data in this mesh using the weights from the node,     |
//  |  and update all dependent data.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMesh::SetPose( const vector<float>& weights )
{
	assert( weights.size() == poses.size() - 1 /* first pose is base pose */ );
	const int weightCount = (int)weights.size();
	// adjust intersection geometry data
	for (int s = (int)vertices.size(), i = 0; i < s; i++)
	{
		vertices[i] = make_float4( poses[0].positions[i], 1 );
		for (int j = 1; j <= weightCount; j++) vertices[i] += weights[j - 1] * make_float4( poses[j].positions[i], 0 );
	}
	// adjust full triangles
	for (int s = (int)triangles.size(), i = 0; i < s; i++)
	{
		triangles[i].vertex0 = make_float3( vertices[i * 3 + 0] );
		triangles[i].vertex1 = make_float3( vertices[i * 3 + 1] );
		triangles[i].vertex2 = make_float3( vertices[i * 3 + 2] );
		triangles[i].vN0 = poses[0].normals[i * 3 + 0];
		triangles[i].vN1 = poses[0].normals[i * 3 + 1];
		triangles[i].vN2 = poses[0].normals[i * 3 + 2];
		for (int j = 1; j <= weightCount; j++)
			triangles[i].vN0 += poses[j].normals[i * 3 + 0],
			triangles[i].vN1 += poses[j].normals[i * 3 + 1],
			triangles[i].vN2 += poses[j].normals[i * 3 + 2];
		triangles[i].vN0 = normalize( triangles[i].vN0 );
		triangles[i].vN1 = normalize( triangles[i].vN1 );
		triangles[i].vN2 = normalize( triangles[i].vN2 );
	}
	// mark as dirty; changing vector contents doesn't trigger this
	MarkAsDirty();
}

//  +-----------------------------------------------------------------------------+
//  |  HostMesh::SetPose                                                          |
//  |  Update the geometry data in this mesh using a skin.                        |
//  |  Called from HostNode::Update, for skinned mesh nodes.                LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMesh::SetPose( const HostSkin* skin, const mat4& meshTransform )
{
	// ensure that we have a backup of the original vertex positions
	if (original.size() == 0)
		for (int s = (int)vertices.size(), i = 0; i < s; i++)
			original.push_back( vertices[i] );
	// transform original into vertex vector using skin matrices
	for (int s = (int)vertices.size(), i = 0; i < s; i++)
	{
		uint4 j4 = joints[i];
		float4 w4 = weights[i];
		mat4 skinMatrix = w4.x * skin->jointMat[j4.x];
		skinMatrix += w4.y * skin->jointMat[j4.y];
		skinMatrix += w4.z * skin->jointMat[j4.z];
		skinMatrix += w4.w * skin->jointMat[j4.w];
        vertices[i] = skinMatrix * original[i];
	}
	// adjust full triangles, TODO: code repetition; TODO: normals
	for (int s = (int)triangles.size(), i = 0; i < s; i++)
	{
		triangles[i].vertex0 = make_float3( vertices[i * 3 + 0] );
		triangles[i].vertex1 = make_float3( vertices[i * 3 + 1] );
		triangles[i].vertex2 = make_float3( vertices[i * 3 + 2] );
	}
	// mark as dirty; changing vector contents doesn't trigger this
	MarkAsDirty();
}

// EOF