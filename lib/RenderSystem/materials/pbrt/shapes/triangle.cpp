
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// shapes/triangle.cpp*

#include "triangle.h"

namespace pbrt
{

HostMesh* CreateTriangleMeshShape(
	const Transform* o2w, const Transform* w2o, bool reverseOrientation,
	const ParamSet& params,
	const int materialIdx,
	std::map<std::string, HostMaterial::ScalarValue*>* floatTextures )
{
	int nvi, npi, nuvi, nsi, nni;
	const int* vi = params.FindInt( "indices", &nvi );
	const Point3f* P = params.FindPoint3f( "P", &npi );
	const Point2f* uvs = params.FindPoint2f( "uv", &nuvi );
	if ( !uvs ) uvs = params.FindPoint2f( "st", &nuvi );
	std::vector<Point2f> tempUVs;
	if ( !uvs )
	{
		const Float* fuv = params.FindFloat( "uv", &nuvi );
		if ( !fuv ) fuv = params.FindFloat( "st", &nuvi );
		if ( fuv )
		{
			nuvi /= 2;
			tempUVs.reserve( nuvi );
			for ( int i = 0; i < nuvi; ++i )
				tempUVs.push_back( {fuv[2 * i], fuv[2 * i + 1]} );
			uvs = tempUVs.data();
		}
	}
	if ( uvs )
	{
		if ( nuvi < npi )
		{
			Error(
				"Not enough of \"uv\"s for triangle mesh.  Expected %d, "
				"found %d.  Discarding.",
				npi, nuvi );
			uvs = nullptr;
		}
		else if ( nuvi > npi )
			Warning(
				"More \"uv\"s provided than will be used for triangle "
				"mesh.  (%d expcted, %d found)",
				npi, nuvi );
	}
	if ( !vi )
	{
		Error(
			"Vertex indices \"indices\" not provided with triangle mesh shape" );
		return nullptr;
	}
	if ( !P )
	{
		Error( "Vertex positions \"P\" not provided with triangle mesh shape" );
		return nullptr;
	}
	const Vector3f* S = params.FindVector3f( "S", &nsi );
	if ( S && nsi != npi )
	{
		Error( "Number of \"S\"s for triangle mesh must match \"P\"s" );
		S = nullptr;
	}
	const Normal3f* N = params.FindNormal3f( "N", &nni );
	if ( N && nni != npi )
	{
		Error( "Number of \"N\"s for triangle mesh must match \"P\"s" );
		N = nullptr;
	}
	for ( int i = 0; i < nvi; ++i )
		if ( vi[i] >= npi )
		{
			Error(
				"trianglemesh has out of-bounds vertex index %d (%d \"P\" "
				"values were given",
				vi[i], npi );
			return nullptr;
		}

	int nfi;
	const int* faceIndices = params.FindInt( "faceIndices", &nfi );
	if ( faceIndices && nfi != nvi / 3 )
	{
		Error( "Number of face indices, %d, doesn't match number of faces, %d",
			   nfi, nvi / 3 );
		faceIndices = nullptr;
	}

	// TODO
#if 0
	std::shared_ptr<Texture<Float>> alphaTex;
	std::string alphaTexName = params.FindTexture( "alpha" );
	if ( alphaTexName != "" )
	{
		if ( floatTextures->find( alphaTexName ) != floatTextures->end() )
			alphaTex = ( *floatTextures )[alphaTexName];
		else
			Error( "Couldn't find float texture \"%s\" for \"alpha\" parameter",
				   alphaTexName.c_str() );
	}
	else if ( params.FindOneFloat( "alpha", 1.f ) == 0.f )
		alphaTex.reset( new ConstantTexture<Float>( 0.f ) );

	std::shared_ptr<Texture<Float>> shadowAlphaTex;
	std::string shadowAlphaTexName = params.FindTexture( "shadowalpha" );
	if ( shadowAlphaTexName != "" )
	{
		if ( floatTextures->find( shadowAlphaTexName ) != floatTextures->end() )
			shadowAlphaTex = ( *floatTextures )[shadowAlphaTexName];
		else
			Error(
				"Couldn't find float texture \"%s\" for \"shadowalpha\" "
				"parameter",
				shadowAlphaTexName.c_str() );
	}
	else if ( params.FindOneFloat( "shadowalpha", 1.f ) == 0.f )
		shadowAlphaTex.reset( new ConstantTexture<Float>( 0.f ) );
#endif

	if ( faceIndices )
		Warning( "faceIndices specified, but not used!" );
	if ( S )
		Warning( "S specified, but not used!" );

	auto mesh = new HostMesh;
	const std::vector<HostMesh::Pose> noPose;
	const std::vector<uint4> noJoints;
	const std::vector<float4> noWeights;

	const std::vector<int> indices( vi, vi + nvi );
	const std::vector<Point3f> vertices( P, P + npi );

	// Optional normals and uvs:
	std::vector<Normal3f> normals;
	if ( N )
		normals = {N, N + nni};
	std::vector<Point2f> uvs_vec;
	if ( uvs )
		uvs_vec = {uvs, uvs + nuvi};

	mesh->BuildFromIndexedData( indices, vertices, normals, uvs_vec, noPose, noJoints, noWeights, materialIdx );

	return mesh;

	// return CreateTriangleMesh( o2w, w2o, reverseOrientation, nvi / 3, vi, npi, P,
	// 						   S, N, uvs, alphaTex, shadowAlphaTex, faceIndices );
}

}; // namespace pbrt
