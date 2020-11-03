
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

#include "materials/pbrt/pbrtparser.h"

#if defined( _MSC_VER )
#pragma warning( disable : 4996 )
#endif

using namespace std;

/* ----------------------------------------------------------------------
 * RPly library, read/write PLY files - Diego Nehab, IMPA
 * http://www.impa.br/~diego/software/rply
 * This library is distributed under the MIT License. See notice
 * at the end of this file.
 * ---------------------------------------------------------------------- */
#define RPLY_VERSION "RPly 1.1.3"
#define RPLY_COPYRIGHT "Copyright (C) 2003-2013 Diego Nehab"
#define RPLY_AUTHORS "Diego Nehab"
typedef struct t_ply_ *p_ply;
typedef struct t_ply_element_ *p_ply_element;
typedef struct t_ply_property_ *p_ply_property;
typedef struct t_ply_argument_ *p_ply_argument;
typedef enum e_ply_storage_mode_
{
	PLY_BIG_ENDIAN, PLY_LITTLE_ENDIAN, PLY_ASCII, PLY_DEFAULT /* has to be the last in enum */
} e_ply_storage_mode; /* order matches ply_storage_mode_list */
typedef enum e_ply_type
{
	PLY_INT8, PLY_UINT8, PLY_INT16, PLY_UINT16,
	PLY_INT32, PLY_UIN32, PLY_FLOAT32, PLY_FLOAT64,
	PLY_CHAR, PLY_UCHAR, PLY_SHORT, PLY_USHORT,
	PLY_INT, PLY_UINT, PLY_FLOAT, PLY_DOUBLE,
	PLY_LIST  /* has to be the last in enum */
} e_ply_type; /* order matches ply_type_list */
typedef void( *p_ply_error_cb )(p_ply ply, const char *message);
p_ply ply_open( const char *name, p_ply_error_cb error_cb, long idata, void *pdata );
int ply_read_header( p_ply ply );
typedef int( *p_ply_read_cb )(p_ply_argument argument);
long ply_set_read_cb( p_ply ply, const char *element_name, const char *property_name, p_ply_read_cb read_cb, void *pdata, long idata );
int ply_get_argument_element( p_ply_argument argument, p_ply_element *element, long *instance_index );
int ply_get_argument_property( p_ply_argument argument, p_ply_property *property, long *length, long *value_index );
int ply_get_argument_user_data( p_ply_argument argument, void **pdata, long *idata );
double ply_get_argument_value( p_ply_argument argument );
int ply_read( p_ply ply );
p_ply_element ply_get_next_element( p_ply ply, p_ply_element last );
int ply_get_element_info( p_ply_element element, const char **name, long *ninstances );
int ply_close( p_ply ply );

namespace pbrt
{

struct CallbackContext
{
	Point3f* p;
	Normal3f* n;
	Point2f* uv;
	int* indices;
	int* faceIndices;
	int indexCtr, faceIndexCtr, face[4];
	bool error;
	int vertexCount;

	CallbackContext()
		: p( nullptr ), n( nullptr ), uv( nullptr ), indices( nullptr ), faceIndices( nullptr ), 
		indexCtr( 0 ), faceIndexCtr( 0 ), error( false ), vertexCount( 0 )
	{
	}
	~CallbackContext()
	{
		delete[] p;
		delete[] n;
		delete[] uv;
		delete[] indices;
		delete[] faceIndices;
	}
};

void rply_message_callback( p_ply ply, const char* message )
{
	Warning( "rply: %s", message );
}

/* Callback to handle vertex data from RPly */
int rply_vertex_callback( p_ply_argument argument )
{
	Float** buffers;
	long index, flags;
	ply_get_argument_user_data( argument, (void**)&buffers, &flags );
	ply_get_argument_element( argument, nullptr, &index );
	int bufferIndex = (flags & 0xF00) >> 8;
	int stride = (flags & 0x0F0) >> 4;
	int offset = flags & 0x00F;
	Float* buffer = buffers[bufferIndex];
	if (buffer) buffer[index * stride + offset] = (float)ply_get_argument_value( argument );
	return 1;
}

/* Callback to handle face data from RPly */
int rply_face_callback( p_ply_argument argument )
{
	CallbackContext* context;
	long flags;
	ply_get_argument_user_data( argument, (void**)&context, &flags );

	if (flags == 0)
	{
		// Vertex indices
		long length, value_index;
		ply_get_argument_property( argument, nullptr, &length, &value_index );

		if (length != 3 && length != 4)
		{
			Warning( "plymesh: Ignoring face with %i vertices (only triangles and quads "
				"are supported!)", (int)length );
			return 1;
		}
		else if (value_index < 0) return 1;
		FATALERROR_IF( length == 4 && context->faceIndices, "face_indices not yet supported for quads" );
		// 	CHECK( context->faceIndices == nullptr ) << "face_indices not yet supported for quads";

		if (value_index >= 0)
		{
			int value = (int)ply_get_argument_value( argument );
			if (value < 0 || value >= context->vertexCount)
			{
				Error( "plymesh: Vertex reference %i is out of bounds! Valid range is [0..%i)",
					value, context->vertexCount );
				context->error = true;
			}
			context->face[value_index] = value;
		}
		if (value_index == length - 1)
		{
			for (int i = 0; i < 3; ++i) context->indices[context->indexCtr++] = context->face[i];
			if (length == 4)
			{
				/* This was a quad */
				context->indices[context->indexCtr++] = context->face[3];
				context->indices[context->indexCtr++] = context->face[0];
				context->indices[context->indexCtr++] = context->face[2];
			}
		}
	}
	else
	{
		CHECK_EQ( 1, flags );
		// Face indices
		context->faceIndices[context->faceIndexCtr++] = (int)ply_get_argument_value( argument );
	}

	return 1;
}

HostMesh* CreatePLYMesh(
	const Transform* o2w, const Transform* w2o, bool reverseOrientation, const ParamSet& params, 
	const int materialIdx, map<string, HostMaterial::ScalarValue*>* floatTextures )
{
	const string filename = params.FindOneFilename( "filename", "" );
	p_ply ply = ply_open( filename.c_str(), rply_message_callback, 0, nullptr );
	if (!ply)
	{
		Error( "Couldn't open PLY file \"%s\"", filename.c_str() );
		return nullptr;
	}
	if (!ply_read_header( ply ))
	{
		Error( "Unable to read the header of PLY file \"%s\"", filename.c_str() );
		return nullptr;
	}

	p_ply_element element = nullptr;
	long vertexCount = 0, faceCount = 0;

	/* Inspect the structure of the PLY file */
	while ((element = ply_get_next_element( ply, element )) != nullptr)
	{
		const char* name;
		long nInstances;
		ply_get_element_info( element, &name, &nInstances );
		if (!strcmp( name, "vertex" )) vertexCount = nInstances;
		else if (!strcmp( name, "face" )) faceCount = nInstances;
	}

	if (vertexCount == 0 || faceCount == 0)
	{
		Error( "%s: PLY file is invalid! No face/vertex elements found!", filename.c_str() );
		return nullptr;
	}

	CallbackContext context;

	if (ply_set_read_cb( ply, "vertex", "x", rply_vertex_callback, &context, 0x030 ) &&
		ply_set_read_cb( ply, "vertex", "y", rply_vertex_callback, &context, 0x031 ) &&
		ply_set_read_cb( ply, "vertex", "z", rply_vertex_callback, &context, 0x032 ))
	{
		context.p = new Point3f[vertexCount];
	}
	else
	{
		Error( "%s: Vertex coordinate property not found!", filename.c_str() );
		return nullptr;
	}

	if (ply_set_read_cb( ply, "vertex", "nx", rply_vertex_callback, &context, 0x130 ) &&
		ply_set_read_cb( ply, "vertex", "ny", rply_vertex_callback, &context, 0x131 ) &&
		ply_set_read_cb( ply, "vertex", "nz", rply_vertex_callback, &context, 0x132 ))
		context.n = new Normal3f[vertexCount];

	/* There seem to be lots of different conventions regarding UV coordinate
	 * names */
	if ((ply_set_read_cb( ply, "vertex", "u", rply_vertex_callback, &context, 0x220 ) &&
		ply_set_read_cb( ply, "vertex", "v", rply_vertex_callback, &context, 0x221 )) ||
			(ply_set_read_cb( ply, "vertex", "s", rply_vertex_callback, &context, 0x220 ) &&
				ply_set_read_cb( ply, "vertex", "t", rply_vertex_callback, &context, 0x221 )) ||
					(ply_set_read_cb( ply, "vertex", "texture_u", rply_vertex_callback, &context, 0x220 ) &&
						ply_set_read_cb( ply, "vertex", "texture_v", rply_vertex_callback, &context, 0x221 )) ||
							(ply_set_read_cb( ply, "vertex", "texture_s", rply_vertex_callback, &context, 0x220 ) &&
								ply_set_read_cb( ply, "vertex", "texture_t", rply_vertex_callback, &context, 0x221 )))
		context.uv = new Point2f[vertexCount];

	/* Allocate enough space in case all faces are quads */
	context.indices = new int[faceCount * 6];
	context.vertexCount = vertexCount;
	ply_set_read_cb( ply, "face", "vertex_indices", rply_face_callback, &context, 0 );
	if (ply_set_read_cb( ply, "face", "face_indices", rply_face_callback, &context, 1 ))
		// Extra space in case they're quads
		context.faceIndices = new int[faceCount];
	if (!ply_read( ply ))
	{
		Error( "%s: unable to read the contents of PLY file", filename.c_str() );
		ply_close( ply );
		return nullptr;
	}
	ply_close( ply );
	if (context.error) return nullptr;

#if 0
	// Look up an alpha texture, if applicable
	HostMaterial::Vec3Value alphaTex;
	string alphaTexName = params.FindTexture( "alpha" );
	if (alphaTexName != "")
	{
		if (floatTextures->find( alphaTexName ) != floatTextures->end()) alphaTex = (*floatTextures)[alphaTexName];
		else Error( "Couldn't find float texture \"%s\" for \"alpha\" parameter", alphaTexName.c_str() );
	}
	else if (params.FindOneFloat( "alpha", 1.f ) == 0.f) alphaTex = new ConstantTexture<Float>( 0.f );

	HostMaterial::Vec3Value shadowAlphaTex;
	string shadowAlphaTexName = params.FindTexture( "shadowalpha" );
	if (shadowAlphaTexName != "")
	{
		if (floatTextures->find( shadowAlphaTexName ) != floatTextures->end())
			shadowAlphaTex = (*floatTextures)[shadowAlphaTexName];
		else Error( "Couldn't find float texture \"%s\" for \"shadowalpha\" parameter",
			shadowAlphaTexName.c_str() );
	}
	else if (params.FindOneFloat( "shadowalpha", 1.f ) == 0.f)
		shadowAlphaTex = new ConstantTexture<Float>( 0.f );
#endif

	auto mesh = new HostMesh;
	const vector<HostMesh::Pose> noPose;
	const vector<uint4> noJoints;
	const vector<float4> noWeights;

	vector<int> indices( context.indices, context.indices + context.indexCtr );
	vector<Point3f> vertices( context.p, context.p + vertexCount );
	vector<Normal3f> normals;
	if (context.n) normals = { context.n, context.n + vertexCount };
	vector<Point2f> uvs, uv2s /* second layer uvs not used for this type of mesh */;
	if (context.uv) uvs = { context.uv, context.uv + vertexCount };
	vector<float4> dummyT;
	mesh->BuildFromIndexedData( indices, vertices, normals, uvs, uv2s, dummyT, noPose, noJoints, noWeights, materialIdx );

	return mesh;
}

HostMesh* CreateTriangleMeshShape(
	const Transform* o2w, const Transform* w2o, bool reverseOrientation,
	const ParamSet& params, const int materialIdx,
	std::map<std::string, HostMaterial::ScalarValue*>* floatTextures )
{
	int nvi, npi, nuvi, nsi, nni;
	const int* vi = params.FindInt( "indices", &nvi );
	const Point3f* P = params.FindPoint3f( "P", &npi );
	const Point2f* uvs = params.FindPoint2f( "uv", &nuvi );
	if (!uvs) uvs = params.FindPoint2f( "st", &nuvi );
	std::vector<Point2f> tempUVs;
	if (!uvs)
	{
		const Float* fuv = params.FindFloat( "uv", &nuvi );
		if (!fuv) fuv = params.FindFloat( "st", &nuvi );
		if (fuv)
		{
			nuvi /= 2;
			tempUVs.reserve( nuvi );
			for (int i = 0; i < nuvi; ++i) tempUVs.push_back( { fuv[2 * i], fuv[2 * i + 1] } );
			uvs = tempUVs.data();
		}
	}
	if (uvs)
	{
		if (nuvi < npi)
		{
			Error( "Not enough of \"uv\"s for triangle mesh. Expected %d, found %d. Discarding.", npi, nuvi );
			uvs = nullptr;
		}
		else if (nuvi > npi)
			Warning( "More \"uv\"s provided than will be used for triangle mesh. (%d expcted, %d found)", npi, nuvi );
	}
	if (!vi)
	{
		Error( "Vertex indices \"indices\" not provided with triangle mesh shape" );
		return nullptr;
	}
	if (!P)
	{
		Error( "Vertex positions \"P\" not provided with triangle mesh shape" );
		return nullptr;
	}
	const Vector3f* S = params.FindVector3f( "S", &nsi );
	if (S && nsi != npi)
	{
		Error( "Number of \"S\"s for triangle mesh must match \"P\"s" );
		S = nullptr;
	}
	const Normal3f* N = params.FindNormal3f( "N", &nni );
	if (N && nni != npi)
	{
		Error( "Number of \"N\"s for triangle mesh must match \"P\"s" );
		N = nullptr;
	}
	for (int i = 0; i < nvi; ++i) if (vi[i] >= npi)
	{
		Error( "trianglemesh has out of-bounds vertex index %d (%d \"P\" values were given", vi[i], npi );
		return nullptr;
	}

	int nfi;
	const int* faceIndices = params.FindInt( "faceIndices", &nfi );
	if (faceIndices && nfi != nvi / 3)
	{
		Error( "Number of face indices, %d, doesn't match number of faces, %d", nfi, nvi / 3 );
		faceIndices = nullptr;
	}

	// TODO
#if 0
	std::shared_ptr<Texture<Float>> alphaTex;
	std::string alphaTexName = params.FindTexture( "alpha" );
	if (alphaTexName != "")
	{
		if (floatTextures->find( alphaTexName ) != floatTextures->end()) alphaTex = (*floatTextures)[alphaTexName];
		else Error( "Couldn't find float texture \"%s\" for \"alpha\" parameter", alphaTexName.c_str() );
	}
	else if (params.FindOneFloat( "alpha", 1.f ) == 0.f)
		alphaTex.reset( new ConstantTexture<Float>( 0.f ) );

	std::shared_ptr<Texture<Float>> shadowAlphaTex;
	std::string shadowAlphaTexName = params.FindTexture( "shadowalpha" );
	if (shadowAlphaTexName != "")
	{
		if (floatTextures->find( shadowAlphaTexName ) != floatTextures->end())
			shadowAlphaTex = (*floatTextures)[shadowAlphaTexName];
		else Error( "Couldn't find float texture \"%s\" for \"shadowalpha\" parameter",
			shadowAlphaTexName.c_str() );
	}
	else if (params.FindOneFloat( "shadowalpha", 1.f ) == 0.f)
		shadowAlphaTex.reset( new ConstantTexture<Float>( 0.f ) );
#endif

	if (faceIndices) Warning( "faceIndices specified, but not used!" );
	if (S) Warning( "S specified, but not used!" );

	auto mesh = new HostMesh;
	const std::vector<HostMesh::Pose> noPose;
	const std::vector<uint4> noJoints;
	const std::vector<float4> noWeights;
	const std::vector<int> indices( vi, vi + nvi );
	const std::vector<Point3f> vertices( P, P + npi );

	// Optional normals and uvs:
	std::vector<Normal3f> normals;
	if (N) normals = { N, N + nni };
	std::vector<Point2f> uvs_vec, uv2s /* second layer uvs not used for this type of mesh */;
	if (uvs) uvs_vec = { uvs, uvs + nuvi };
	vector<float4> dummyT;
	mesh->BuildFromIndexedData( indices, vertices, normals, uvs_vec, uv2s, dummyT, noPose, noJoints, noWeights, materialIdx );
	return mesh;
}

} // namespace pbrt

/* ----------------------------------------------------------------------
 * RPly library, read/write PLY files
 * Diego Nehab, IMPA
 * http://www.impa.br/~diego/software/rply
 *
 * This library is distributed under the MIT License. See notice
 * at the end of this file.
 * ---------------------------------------------------------------------- */

#include "materials/pbrt/pbrtparser.h"

 /* ----------------------------------------------------------------------
  * Make sure we get our integer types right
  * ---------------------------------------------------------------------- */
#if defined(_MSC_VER) && (_MSC_VER < 1600)
  /* C99 stdint.h only supported in MSVC++ 10.0 and up */
typedef __int8 t_ply_int8;
typedef __int16 t_ply_int16;
typedef __int32 t_ply_int32;
typedef unsigned __int8 t_ply_uint8;
typedef unsigned __int16 t_ply_uint16;
typedef unsigned __int32 t_ply_uint32;
#define PLY_INT8_MAX (127)
#define PLY_INT8_MIN (-PLY_INT8_MAX - 1)
#define PLY_INT16_MAX (32767)
#define PLY_INT16_MIN (-PLY_INT16_MAX - 1)
#define PLY_INT32_MAX (2147483647)
#define PLY_INT32_MIN (-PLY_INT32_MAX - 1)
#define PLY_UINT8_MAX (255)
#define PLY_UINT16_MAX (65535)
#define PLY_UINT32_MAX (4294967295)
#else
#define __STDC_LIMIT_MACROS
#include <stdint.h>
typedef int8_t t_ply_int8;
typedef int16_t t_ply_int16;
typedef int32_t t_ply_int32;
typedef uint8_t t_ply_uint8;
typedef uint16_t t_ply_uint16;
typedef uint32_t t_ply_uint32;
#define PLY_INT8_MIN INT8_MIN
#define PLY_INT8_MAX INT8_MAX
#define PLY_INT16_MIN INT16_MIN
#define PLY_INT16_MAX INT16_MAX
#define PLY_INT32_MIN INT32_MIN
#define PLY_INT32_MAX INT32_MAX
#define PLY_UINT8_MAX UINT8_MAX
#define PLY_UINT16_MAX UINT16_MAX
#define PLY_UINT32_MAX UINT32_MAX
#endif

/* ----------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */
#define WORDSIZE 256
#define LINESIZE 1024
#define BUFFERSIZE (8 * 1024)

typedef enum e_ply_io_mode_ { PLY_READ, PLY_WRITE } e_ply_io_mode;

static const char *const ply_storage_mode_list[] = {
	"binary_big_endian", "binary_little_endian", "ascii",
	NULL }; /* order matches e_ply_storage_mode enum */

static const char *const ply_type_list[] = {
	"int8",    "uint8",  "int16", "uint16", "int32",  "uint32", "float32",
	"float64", "char",   "uchar", "short",  "ushort", "int",    "uint",
	"float",   "double", "list",  NULL }; /* order matches e_ply_type enum */

/* ----------------------------------------------------------------------
 * Property reading callback argument
 *
 * element: name of element being processed
 * property: name of property being processed
 * nelements: number of elements of this kind in file
 * instance_index: index current element of this kind being processed
 * length: number of values in current list (or 1 for scalars)
 * value_index: index of current value int this list (or 0 for scalars)
 * value: value of property
 * pdata/idata: user data defined with ply_set_cb
 *
 * Returns handle to PLY file if succesful, NULL otherwise.
 * ---------------------------------------------------------------------- */
typedef struct t_ply_argument_
{
	p_ply_element element;
	long instance_index;
	p_ply_property property;
	long length, value_index;
	double value;
	void *pdata;
	long idata;
} t_ply_argument;

/* ----------------------------------------------------------------------
 * Property information
 *
 * name: name of this property
 * type: type of this property (list or type of scalar value)
 * length_type, value_type: type of list property count and values
 * read_cb: function to be called when this property is called
 *
 * Returns 1 if should continue processing file, 0 if should abort.
 * ---------------------------------------------------------------------- */
typedef struct t_ply_property_
{
	char name[WORDSIZE];
	e_ply_type type, value_type, length_type;
	p_ply_read_cb read_cb;
	void *pdata;
	long idata;
} t_ply_property;

/* ----------------------------------------------------------------------
 * Element information
 *
 * name: name of this property
 * ninstances: number of elements of this type in file
 * property: property descriptions for this element
 * nproperty: number of properties in this element
 *
 * Returns 1 if should continue processing file, 0 if should abort.
 * ---------------------------------------------------------------------- */
typedef struct t_ply_element_
{
	char name[WORDSIZE];
	long ninstances;
	p_ply_property property;
	long nproperties;
} t_ply_element;

/* ----------------------------------------------------------------------
 * Input/output driver
 *
 * Depending on file mode, different functions are used to read/write
 * property fields. The drivers make it transparent to read/write in ascii,
 * big endian or little endian cases.
 * ---------------------------------------------------------------------- */
typedef int( *p_ply_ihandler )(p_ply ply, double *value);
typedef int( *p_ply_ichunk )(p_ply ply, void *anydata, size_t size);
typedef struct t_ply_idriver_
{
	p_ply_ihandler ihandler[16];
	p_ply_ichunk ichunk;
	const char *name;
} t_ply_idriver;
typedef t_ply_idriver *p_ply_idriver;

typedef int( *p_ply_ohandler )(p_ply ply, double value);
typedef int( *p_ply_ochunk )(p_ply ply, void *anydata, size_t size);
typedef struct t_ply_odriver_
{
	p_ply_ohandler ohandler[16];
	p_ply_ochunk ochunk;
	const char *name;
} t_ply_odriver;
typedef t_ply_odriver *p_ply_odriver;

/* ----------------------------------------------------------------------
 * Ply file handle.
 *
 * io_mode: read or write (from e_ply_io_mode)
 * storage_mode: mode of file associated with handle (from e_ply_storage_mode)
 * element: elements description for this file
 * nelement: number of different elements in file
 * comment: comments for this file
 * ncomments: number of comments in file
 * obj_info: obj_info items for this file
 * nobj_infos: number of obj_info items in file
 * fp: file pointer associated with ply file
 * rn: skip extra char after end_header?
 * buffer: last word/chunck of data read from ply file
 * buffer_first, buffer_last: interval of untouched good data in buffer
 * buffer_token: start of parsed token (line or word) in buffer
 * idriver, odriver: input driver used to get property fields from file
 * argument: storage space for callback arguments
 * welement, wproperty: element/property type being written
 * winstance_index: index of instance of current element being written
 * wvalue_index: index of list property value being written
 * wlength: number of values in list property being written
 * error_cb: error callback
 * pdata/idata: user data defined with ply_open/ply_create
 * ---------------------------------------------------------------------- */
typedef struct t_ply_
{
	e_ply_io_mode io_mode;
	e_ply_storage_mode storage_mode;
	p_ply_element element;
	long nelements;
	char *comment;
	long ncomments;
	char *obj_info;
	long nobj_infos;
	FILE *fp;
	int rn;
	char buffer[BUFFERSIZE];
	size_t buffer_first, buffer_token, buffer_last;
	p_ply_idriver idriver;
	p_ply_odriver odriver;
	t_ply_argument argument;
	long welement, wproperty;
	long winstance_index, wvalue_index, wlength;
	p_ply_error_cb error_cb;
	void *pdata;
	long idata;
} t_ply;

/* ----------------------------------------------------------------------
 * I/O functions and drivers
 * ---------------------------------------------------------------------- */
namespace
{
extern t_ply_idriver ply_idriver_ascii;
extern t_ply_idriver ply_idriver_binary;
extern t_ply_idriver ply_idriver_binary_reverse;
extern t_ply_odriver ply_odriver_ascii;
extern t_ply_odriver ply_odriver_binary;
extern t_ply_odriver ply_odriver_binary_reverse;
};

static int ply_read_word( p_ply ply );
static int ply_check_word( p_ply ply );
static void ply_finish_word( p_ply ply, size_t size );
static int ply_read_line( p_ply ply );
static int ply_check_line( p_ply ply );
static int ply_read_chunk( p_ply ply, void *anybuffer, size_t size );
static int ply_read_chunk_reverse( p_ply ply, void *anybuffer, size_t size );
static void ply_reverse( void *anydata, size_t size );

/* ----------------------------------------------------------------------
 * String functions
 * ---------------------------------------------------------------------- */
static int ply_find_string( const char *item, const char *const list[] );
static p_ply_element ply_find_element( p_ply ply, const char *name );
static p_ply_property ply_find_property( p_ply_element element,
	const char *name );

/* ----------------------------------------------------------------------
 * Header parsing
 * ---------------------------------------------------------------------- */
static int ply_read_header_magic( p_ply ply );
static int ply_read_header_format( p_ply ply );
static int ply_read_header_comment( p_ply ply );
static int ply_read_header_obj_info( p_ply ply );
static int ply_read_header_property( p_ply ply );
static int ply_read_header_element( p_ply ply );

/* ----------------------------------------------------------------------
 * Error handling
 * ---------------------------------------------------------------------- */
static void ply_error_cb( p_ply ply, const char *message );
static void ply_ferror( p_ply ply, const char *fmt, ... );

/* ----------------------------------------------------------------------
 * Memory allocation and initialization
 * ---------------------------------------------------------------------- */
static void ply_init( p_ply ply );
static void ply_element_init( p_ply_element element );
static void ply_property_init( p_ply_property property );
static p_ply ply_alloc( void );
static p_ply_element ply_grow_element( p_ply ply );
static p_ply_property ply_grow_property( p_ply ply, p_ply_element element );
static void *ply_grow_array( p_ply ply, void **pointer, long *nmemb, long size );

/* ----------------------------------------------------------------------
 * Special functions
 * ---------------------------------------------------------------------- */
static e_ply_storage_mode ply_arch_endian( void );
static int ply_type_check( void );

/* ----------------------------------------------------------------------
 * Auxiliary read functions
 * ---------------------------------------------------------------------- */
static int ply_read_element( p_ply ply, p_ply_element element,
	p_ply_argument argument );
static int ply_read_property( p_ply ply, p_ply_element element,
	p_ply_property property, p_ply_argument argument );
static int ply_read_list_property( p_ply ply, p_ply_element element,
	p_ply_property property,
	p_ply_argument argument );
static int ply_read_scalar_property( p_ply ply, p_ply_element element,
	p_ply_property property,
	p_ply_argument argument );

/* ----------------------------------------------------------------------
 * Buffer support functions
 * ---------------------------------------------------------------------- */
 /* pointers to tokenized word and line in buffer */
#define BWORD(p) (p->buffer + p->buffer_token)
#define BLINE(p) (p->buffer + p->buffer_token)

/* pointer to start of untouched bytes in buffer */
#define BFIRST(p) (p->buffer + p->buffer_first)

/* number of bytes untouched in buffer */
#define BSIZE(p) (p->buffer_last - p->buffer_first)

/* consumes data from buffer */
#define BSKIP(p, s) (p->buffer_first += s)

/* refills the buffer */
static int BREFILL( p_ply ply )
{
	/* move untouched data to beginning of buffer */
	size_t size = BSIZE( ply );
	memmove( ply->buffer, BFIRST( ply ), size );
	ply->buffer_last = size;
	ply->buffer_first = ply->buffer_token = 0;
	/* fill remaining with new data */
	size = fread( ply->buffer + size, 1, BUFFERSIZE - size - 1, ply->fp );
	/* place sentinel so we can use str* functions with buffer */
	ply->buffer[BUFFERSIZE - 1] = '\0';
	/* check if read failed */
	if (size <= 0) return 0;
	/* increase size to account for new data */
	ply->buffer_last += size;
	return 1;
}

/* We don't care about end-of-line, generally, because we
 * separate words by any white-space character.
 * Unfortunately, in binary mode, right after 'end_header',
 * we have to know *exactly* how many characters to skip */
 /* We use the end-of-line marker after the 'ply' magic
  * number to figure out what to do */
static int ply_read_header_magic( p_ply ply )
{
	char *magic = ply->buffer;
	if (!BREFILL( ply ))
	{
		ply->error_cb( ply, "Unable to read magic number from file" );
		return 0;
	}
	/* check if it is ply */
	if (magic[0] != 'p' || magic[1] != 'l' || magic[2] != 'y' ||
		!isspace( magic[3] ))
	{
		ply->error_cb( ply, "Wrong magic number. Expected 'ply'" );
		return 0;
	}
	/* figure out if we have to skip the extra character
	 * after header when we reach the binary part of file */
	ply->rn = magic[3] == '\r' && magic[4] == '\n';
	BSKIP( ply, 3 );
	return 1;
}

/* ----------------------------------------------------------------------
 * Exported functions
 * ---------------------------------------------------------------------- */
 /* ----------------------------------------------------------------------
  * Read support functions
  * ---------------------------------------------------------------------- */
p_ply ply_open( const char *name, p_ply_error_cb error_cb, long idata,
	void *pdata )
{
	FILE *fp = NULL;
	p_ply ply = ply_alloc();
	if (error_cb == NULL) error_cb = ply_error_cb;
	if (!ply)
	{
		error_cb( NULL, "Out of memory" );
		return NULL;
	}
	ply->idata = idata;
	ply->pdata = pdata;
	ply->io_mode = PLY_READ;
	ply->error_cb = error_cb;
	if (!ply_type_check())
	{
		error_cb( ply, "Incompatible type system" );
		free( ply );
		return NULL;
	}
	assert( name );
	fp = fopen( name, "rb" );
	if (!fp)
	{
		error_cb( ply, "Unable to open file" );
		free( ply );
		return NULL;
	}
	ply->fp = fp;
	return ply;
}

int ply_read_header( p_ply ply )
{
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	if (!ply_read_header_magic( ply )) return 0;
	if (!ply_read_word( ply )) return 0;
	/* parse file format */
	if (!ply_read_header_format( ply ))
	{
		ply_ferror( ply, "Invalid file format" );
		return 0;
	}
	/* parse elements, comments or obj_infos until the end of header */
	while (strcmp( BWORD( ply ), "end_header" ))
	{
		if (!ply_read_header_comment( ply ) && !ply_read_header_element( ply ) &&
			!ply_read_header_obj_info( ply ))
		{
			ply_ferror( ply, "Unexpected token '%s'", BWORD( ply ) );
			return 0;
		}
	}
	/* skip extra character? */
	if (ply->rn)
	{
		if (BSIZE( ply ) < 1 && !BREFILL( ply ))
		{
			ply_ferror( ply, "Unexpected end of file" );
			return 0;
		}
		BSKIP( ply, 1 );
	}
	return 1;
}

long ply_set_read_cb( p_ply ply, const char *element_name,
	const char *property_name, p_ply_read_cb read_cb,
	void *pdata, long idata )
{
	p_ply_element element = NULL;
	p_ply_property property = NULL;
	assert( ply && element_name && property_name );
	element = ply_find_element( ply, element_name );
	if (!element) return 0;
	property = ply_find_property( element, property_name );
	if (!property) return 0;
	property->read_cb = read_cb;
	property->pdata = pdata;
	property->idata = idata;
	return (int)element->ninstances;
}

int ply_read( p_ply ply )
{
	long i;
	p_ply_argument argument;
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	argument = &ply->argument;
	/* for each element type */
	for (i = 0; i < ply->nelements; i++)
	{
		p_ply_element element = &ply->element[i];
		argument->element = element;
		if (!ply_read_element( ply, element, argument )) return 0;
	}
	return 1;
}

int ply_close( p_ply ply )
{
	long i;
	assert( ply && ply->fp );
	assert( ply->element || ply->nelements == 0 );
	assert( !ply->element || ply->nelements > 0 );
	/* write last chunk to file */
	if (ply->io_mode == PLY_WRITE &&
		fwrite( ply->buffer, 1, ply->buffer_last, ply->fp ) < ply->buffer_last)
	{
		ply_ferror( ply, "Error closing up" );
		return 0;
	}
	fclose( ply->fp );
	/* free all memory used by handle */
	if (ply->element)
	{
		for (i = 0; i < ply->nelements; i++)
		{
			p_ply_element element = &ply->element[i];
			if (element->property) free( element->property );
		}
		free( ply->element );
	}
	if (ply->obj_info) free( ply->obj_info );
	if (ply->comment) free( ply->comment );
	free( ply );
	return 1;
}

/* ----------------------------------------------------------------------
 * Query support functions
 * ---------------------------------------------------------------------- */
p_ply_element ply_get_next_element( p_ply ply, p_ply_element last )
{
	assert( ply );
	if (!last) return ply->element;
	last++;
	if (last < ply->element + ply->nelements)
		return last;
	else
		return NULL;
}

int ply_get_element_info( p_ply_element element, const char **name,
	long *ninstances )
{
	assert( element );
	if (name) *name = element->name;
	if (ninstances) *ninstances = (long)element->ninstances;
	return 1;
}

/* ----------------------------------------------------------------------
 * Callback argument support functions
 * ---------------------------------------------------------------------- */
int ply_get_argument_element( p_ply_argument argument, p_ply_element *element,
	long *instance_index )
{
	assert( argument );
	if (!argument) return 0;
	if (element) *element = argument->element;
	if (instance_index) *instance_index = argument->instance_index;
	return 1;
}

int ply_get_argument_property( p_ply_argument argument, p_ply_property *property,
	long *length, long *value_index )
{
	assert( argument );
	if (!argument) return 0;
	if (property) *property = argument->property;
	if (length) *length = argument->length;
	if (value_index) *value_index = argument->value_index;
	return 1;
}

int ply_get_argument_user_data( p_ply_argument argument, void **pdata,
	long *idata )
{
	assert( argument );
	if (!argument) return 0;
	if (pdata) *pdata = argument->pdata;
	if (idata) *idata = argument->idata;
	return 1;
}

double ply_get_argument_value( p_ply_argument argument )
{
	assert( argument );
	if (!argument) return 0.0;
	return argument->value;
}

int ply_add_comment( p_ply ply, const char *comment )
{
	char *new_comment = NULL;
	assert( ply && comment && strlen( comment ) < LINESIZE );
	if (!comment || strlen( comment ) >= LINESIZE)
	{
		ply_ferror( ply, "Invalid arguments" );
		return 0;
	}
	new_comment = (char *)ply_grow_array( ply, (void **)&ply->comment,
		&ply->ncomments, LINESIZE );
	if (!new_comment) return 0;
	strcpy( new_comment, comment );
	return 1;
}

int ply_add_obj_info( p_ply ply, const char *obj_info )
{
	char *new_obj_info = NULL;
	assert( ply && obj_info && strlen( obj_info ) < LINESIZE );
	if (!obj_info || strlen( obj_info ) >= LINESIZE)
	{
		ply_ferror( ply, "Invalid arguments" );
		return 0;
	}
	new_obj_info = (char *)ply_grow_array( ply, (void **)&ply->obj_info,
		&ply->nobj_infos, LINESIZE );
	if (!new_obj_info) return 0;
	strcpy( new_obj_info, obj_info );
	return 1;
}

/* ----------------------------------------------------------------------
 * Internal functions
 * ---------------------------------------------------------------------- */
static int ply_read_list_property( p_ply ply, p_ply_element element,
	p_ply_property property,
	p_ply_argument argument )
{
	int l;
	p_ply_read_cb read_cb = property->read_cb;
	p_ply_ihandler *driver = ply->idriver->ihandler;
	/* get list length */
	p_ply_ihandler handler = driver[property->length_type];
	double length;
	if (!handler( ply, &length ))
	{
		ply_ferror( ply, "Error reading '%s' of '%s' number %d", property->name,
			element->name, argument->instance_index );
		return 0;
	}
	/* invoke callback to pass length in value field */
	argument->length = (long)length;
	argument->value_index = -1;
	argument->value = length;
	if (read_cb && !read_cb( argument ))
	{
		ply_ferror( ply, "Aborted by user" );
		return 0;
	}
	/* read list values */
	handler = driver[property->value_type];
	/* for each value in list */
	for (l = 0; l < (long)length; l++)
	{
		/* read value from file */
		argument->value_index = l;
		if (!handler( ply, &argument->value ))
		{
			ply_ferror( ply,
				"Error reading value number %d of '%s' of "
				"'%s' number %d",
				l + 1, property->name, element->name,
				argument->instance_index );
			return 0;
		}
		/* invoke callback to pass value */
		if (read_cb && !read_cb( argument ))
		{
			ply_ferror( ply, "Aborted by user" );
			return 0;
		}
	}
	return 1;
}

static int ply_read_scalar_property( p_ply ply, p_ply_element element,
	p_ply_property property,
	p_ply_argument argument )
{
	p_ply_read_cb read_cb = property->read_cb;
	p_ply_ihandler *driver = ply->idriver->ihandler;
	p_ply_ihandler handler = driver[property->type];
	argument->length = 1;
	argument->value_index = 0;
	if (!handler( ply, &argument->value ))
	{
		ply_ferror( ply, "Error reading '%s' of '%s' number %d", property->name,
			element->name, argument->instance_index );
		return 0;
	}
	if (read_cb && !read_cb( argument ))
	{
		ply_ferror( ply, "Aborted by user" );
		return 0;
	}
	return 1;
}

static int ply_read_property( p_ply ply, p_ply_element element,
	p_ply_property property, p_ply_argument argument )
{
	if (property->type == PLY_LIST)
		return ply_read_list_property( ply, element, property, argument );
	else
		return ply_read_scalar_property( ply, element, property, argument );
}

static int ply_read_element( p_ply ply, p_ply_element element,
	p_ply_argument argument )
{
	long j, k;
	/* for each element of this type */
	for (j = 0; j < element->ninstances; j++)
	{
		argument->instance_index = j;
		/* for each property */
		for (k = 0; k < element->nproperties; k++)
		{
			p_ply_property property = &element->property[k];
			argument->property = property;
			argument->pdata = property->pdata;
			argument->idata = property->idata;
			if (!ply_read_property( ply, element, property, argument )) return 0;
		}
	}
	return 1;
}

static int ply_find_string( const char *item, const char *const list[] )
{
	int i;
	assert( item && list );
	for (i = 0; list[i]; i++)
		if (!strcmp( list[i], item )) return i;
	return -1;
}

static p_ply_element ply_find_element( p_ply ply, const char *name )
{
	p_ply_element element;
	int i, nelements;
	assert( ply && name );
	element = ply->element;
	nelements = ply->nelements;
	assert( element || nelements == 0 );
	assert( !element || nelements > 0 );
	for (i = 0; i < nelements; i++)
		if (!strcmp( element[i].name, name )) return &element[i];
	return NULL;
}

static p_ply_property ply_find_property( p_ply_element element,
	const char *name )
{
	p_ply_property property;
	int i, nproperties;
	assert( element && name );
	property = element->property;
	nproperties = element->nproperties;
	assert( property || nproperties == 0 );
	assert( !property || nproperties > 0 );
	for (i = 0; i < nproperties; i++)
		if (!strcmp( property[i].name, name )) return &property[i];
	return NULL;
}

static int ply_check_word( p_ply ply )
{
	size_t size = strlen( BWORD( ply ) );
	if (size >= WORDSIZE)
	{
		ply_ferror( ply, "Word too long" );
		return 0;
	}
	else if (size == 0)
	{
		ply_ferror( ply, "Unexpected end of file" );
		return 0;
	}
	return 1;
}

static int ply_read_word( p_ply ply )
{
	size_t t = 0;
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	/* skip leading blanks */
	while (1)
	{
		t = strspn( BFIRST( ply ), " \n\r\t" );
		/* check if all buffer was made of blanks */
		if (t >= BSIZE( ply ))
		{
			if (!BREFILL( ply ))
			{
				ply_ferror( ply, "Unexpected end of file" );
				return 0;
			}
		}
		else
			break;
	}
	BSKIP( ply, t );
	/* look for a space after the current word */
	t = strcspn( BFIRST( ply ), " \n\r\t" );
	/* if we didn't reach the end of the buffer, we are done */
	if (t < BSIZE( ply ))
	{
		ply_finish_word( ply, t );
		return ply_check_word( ply );
	}
	/* otherwise, try to refill buffer */
	if (!BREFILL( ply ))
	{
		/* if we reached the end of file, try to do with what we have */
		ply_finish_word( ply, t );
		return ply_check_word( ply );
		/* ply_ferror(ply, "Unexpected end of file"); */
		/* return 0; */
	}
	/* keep looking from where we left */
	t += strcspn( BFIRST( ply ) + t, " \n\r\t" );
	/* check if the token is too large for our buffer */
	if (t >= BSIZE( ply ))
	{
		ply_ferror( ply, "Token too large" );
		return 0;
	}
	/* we are done */
	ply_finish_word( ply, t );
	return ply_check_word( ply );
}

static void ply_finish_word( p_ply ply, size_t size )
{
	ply->buffer_token = ply->buffer_first;
	BSKIP( ply, size );
	*BFIRST( ply ) = '\0';
	BSKIP( ply, 1 );
}

static int ply_check_line( p_ply ply )
{
	if (strlen( BLINE( ply ) ) >= LINESIZE)
	{
		ply_ferror( ply, "Line too long" );
		return 0;
	}
	return 1;
}

static int ply_read_line( p_ply ply )
{
	const char *end = NULL;
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	/* look for a end of line */
	end = strchr( BFIRST( ply ), '\n' );
	/* if we didn't reach the end of the buffer, we are done */
	if (end)
	{
		ply->buffer_token = ply->buffer_first;
		BSKIP( ply, end - BFIRST( ply ) );
		*BFIRST( ply ) = '\0';
		BSKIP( ply, 1 );
		return ply_check_line( ply );
	}
	else
	{
		end = ply->buffer + BSIZE( ply );
		/* otherwise, try to refill buffer */
		if (!BREFILL( ply ))
		{
			ply_ferror( ply, "Unexpected end of file" );
			return 0;
		}
	}
	/* keep looking from where we left */
	end = strchr( end, '\n' );
	/* check if the token is too large for our buffer */
	if (!end)
	{
		ply_ferror( ply, "Token too large" );
		return 0;
	}
	/* we are done */
	ply->buffer_token = ply->buffer_first;
	BSKIP( ply, end - BFIRST( ply ) );
	*BFIRST( ply ) = '\0';
	BSKIP( ply, 1 );
	return ply_check_line( ply );
}

static int ply_read_chunk( p_ply ply, void *anybuffer, size_t size )
{
	char *buffer = (char *)anybuffer;
	size_t i = 0;
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	assert( ply->buffer_first <= ply->buffer_last );
	while (i < size)
	{
		if (ply->buffer_first < ply->buffer_last)
		{
			buffer[i] = ply->buffer[ply->buffer_first];
			ply->buffer_first++;
			i++;
		}
		else
		{
			ply->buffer_first = 0;
			ply->buffer_last = fread( ply->buffer, 1, BUFFERSIZE, ply->fp );
			if (ply->buffer_last <= 0) return 0;
		}
	}
	return 1;
}

static int ply_read_chunk_reverse( p_ply ply, void *anybuffer, size_t size )
{
	if (!ply_read_chunk( ply, anybuffer, size )) return 0;
	ply_reverse( anybuffer, size );
	return 1;
}

static void ply_reverse( void *anydata, size_t size )
{
	char *data = (char *)anydata;
	char temp;
	size_t i;
	for (i = 0; i < size / 2; i++)
	{
		temp = data[i];
		data[i] = data[size - i - 1];
		data[size - i - 1] = temp;
	}
}

static void ply_init( p_ply ply )
{
	ply->element = NULL;
	ply->nelements = 0;
	ply->comment = NULL;
	ply->ncomments = 0;
	ply->obj_info = NULL;
	ply->nobj_infos = 0;
	ply->idriver = NULL;
	ply->odriver = NULL;
	ply->buffer[0] = '\0';
	ply->buffer_first = ply->buffer_last = ply->buffer_token = 0;
	ply->welement = 0;
	ply->wproperty = 0;
	ply->winstance_index = 0;
	ply->wlength = 0;
	ply->wvalue_index = 0;
}

static void ply_element_init( p_ply_element element )
{
	element->name[0] = '\0';
	element->ninstances = 0;
	element->property = NULL;
	element->nproperties = 0;
}

static void ply_property_init( p_ply_property property )
{
	property->name[0] = '\0';
	property->type = (e_ply_type)-1;
	property->length_type = (e_ply_type)-1;
	property->value_type = (e_ply_type)-1;
	property->read_cb = (p_ply_read_cb)NULL;
	property->pdata = NULL;
	property->idata = 0;
}

static p_ply ply_alloc( void )
{
	p_ply ply = (p_ply)calloc( 1, sizeof( t_ply ) );
	if (!ply) return NULL;
	ply_init( ply );
	return ply;
}

static void *ply_grow_array( p_ply ply, void **pointer, long *nmemb, long size )
{
	void *temp = *pointer;
	long count = *nmemb + 1;
	if (!temp)
		temp = malloc( count * size );
	else
		temp = realloc( temp, count * size );
	if (!temp)
	{
		ply_ferror( ply, "Out of memory" );
		return NULL;
	}
	*pointer = temp;
	*nmemb = count;
	return (char *)temp + (count - 1) * size;
}

static p_ply_element ply_grow_element( p_ply ply )
{
	p_ply_element element = NULL;
	assert( ply );
	assert( ply->element || ply->nelements == 0 );
	assert( !ply->element || ply->nelements > 0 );
	element = (p_ply_element)ply_grow_array(
		ply, (void **)&ply->element, &ply->nelements, sizeof( t_ply_element ) );
	if (!element) return NULL;
	ply_element_init( element );
	return element;
}

static p_ply_property ply_grow_property( p_ply ply, p_ply_element element )
{
	p_ply_property property = NULL;
	assert( ply );
	assert( element );
	assert( element->property || element->nproperties == 0 );
	assert( !element->property || element->nproperties > 0 );
	property = (p_ply_property)ply_grow_array( ply, (void **)&element->property,
		&element->nproperties,
		sizeof( t_ply_property ) );
	if (!property) return NULL;
	ply_property_init( property );
	return property;
}

static int ply_read_header_format( p_ply ply )
{
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	if (strcmp( BWORD( ply ), "format" )) return 0;
	if (!ply_read_word( ply )) return 0;
	ply->storage_mode =
		(e_ply_storage_mode)ply_find_string( BWORD( ply ), ply_storage_mode_list );
	if (ply->storage_mode == (e_ply_storage_mode)(-1)) return 0;
	if (ply->storage_mode == PLY_ASCII)
		ply->idriver = &ply_idriver_ascii;
	else if (ply->storage_mode == ply_arch_endian())
		ply->idriver = &ply_idriver_binary;
	else
		ply->idriver = &ply_idriver_binary_reverse;
	if (!ply_read_word( ply )) return 0;
	if (strcmp( BWORD( ply ), "1.0" )) return 0;
	if (!ply_read_word( ply )) return 0;
	return 1;
}

static int ply_read_header_comment( p_ply ply )
{
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	if (strcmp( BWORD( ply ), "comment" )) return 0;
	if (!ply_read_line( ply )) return 0;
	if (!ply_add_comment( ply, BLINE( ply ) )) return 0;
	if (!ply_read_word( ply )) return 0;
	return 1;
}

static int ply_read_header_obj_info( p_ply ply )
{
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	if (strcmp( BWORD( ply ), "obj_info" )) return 0;
	if (!ply_read_line( ply )) return 0;
	if (!ply_add_obj_info( ply, BLINE( ply ) )) return 0;
	if (!ply_read_word( ply )) return 0;
	return 1;
}

static int ply_read_header_property( p_ply ply )
{
	p_ply_element element = NULL;
	p_ply_property property = NULL;
	/* make sure it is a property */
	if (strcmp( BWORD( ply ), "property" )) return 0;
	element = &ply->element[ply->nelements - 1];
	property = ply_grow_property( ply, element );
	if (!property) return 0;
	/* get property type */
	if (!ply_read_word( ply )) return 0;
	property->type = (e_ply_type)ply_find_string( BWORD( ply ), ply_type_list );
	if (property->type == (e_ply_type)(-1)) return 0;
	if (property->type == PLY_LIST)
	{
		/* if it's a list, we need the base types */
		if (!ply_read_word( ply )) return 0;
		property->length_type =
			(e_ply_type)ply_find_string( BWORD( ply ), ply_type_list );
		if (property->length_type == (e_ply_type)(-1)) return 0;
		if (!ply_read_word( ply )) return 0;
		property->value_type =
			(e_ply_type)ply_find_string( BWORD( ply ), ply_type_list );
		if (property->value_type == (e_ply_type)(-1)) return 0;
	}
	/* get property name */
	if (!ply_read_word( ply )) return 0;
	strcpy( property->name, BWORD( ply ) );
	if (!ply_read_word( ply )) return 0;
	return 1;
}

static int ply_read_header_element( p_ply ply )
{
	p_ply_element element = NULL;
	long dummy;
	assert( ply && ply->fp && ply->io_mode == PLY_READ );
	if (strcmp( BWORD( ply ), "element" )) return 0;
	/* allocate room for new element */
	element = ply_grow_element( ply );
	if (!element) return 0;
	/* get element name */
	if (!ply_read_word( ply )) return 0;
	strcpy( element->name, BWORD( ply ) );
	/* get number of elements of this type */
	if (!ply_read_word( ply )) return 0;
	if (sscanf( BWORD( ply ), "%ld", &dummy ) != 1)
	{
		ply_ferror( ply, "Expected number got '%s'", BWORD( ply ) );
		return 0;
	}
	element->ninstances = dummy;
	/* get all properties for this element */
	if (!ply_read_word( ply )) return 0;
	while (ply_read_header_property( ply ) || ply_read_header_comment( ply ) || ply_read_header_obj_info( ply )) /* do nothing */;
	return 1;
}

static void ply_error_cb( p_ply ply, const char *message )
{
	(void)ply;
	fprintf( stderr, "RPly: %s\n", message );
}

static void ply_ferror( p_ply ply, const char *fmt, ... )
{
	char buffer[1024];
	va_list ap;
	va_start( ap, fmt );
	vsprintf( buffer, fmt, ap );
	va_end( ap );
	ply->error_cb( ply, buffer );
}

static e_ply_storage_mode ply_arch_endian( void )
{
	unsigned long i = 1;
	unsigned char *s = (unsigned char *)&i;
	if (*s == 1) return PLY_LITTLE_ENDIAN; else return PLY_BIG_ENDIAN;
}

static int ply_type_check( void )
{
	assert( sizeof( t_ply_int8 ) == 1 );
	assert( sizeof( t_ply_uint8 ) == 1 );
	assert( sizeof( t_ply_int16 ) == 2 );
	assert( sizeof( t_ply_uint16 ) == 2 );
	assert( sizeof( t_ply_int32 ) == 4 );
	assert( sizeof( t_ply_uint32 ) == 4 );
	assert( sizeof( float ) == 4 );
	assert( sizeof( double ) == 8 );
	if (sizeof( t_ply_int8 ) != 1) return 0;
	if (sizeof( t_ply_uint8 ) != 1) return 0;
	if (sizeof( t_ply_int16 ) != 2) return 0;
	if (sizeof( t_ply_uint16 ) != 2) return 0;
	if (sizeof( t_ply_int32 ) != 4) return 0;
	if (sizeof( t_ply_uint32 ) != 4) return 0;
	if (sizeof( float ) != 4) return 0;
	if (sizeof( double ) != 8) return 0;
	return 1;
}

/* ----------------------------------------------------------------------
 * Output handlers
 * ---------------------------------------------------------------------- */
static int oascii_int8( p_ply ply, double value )
{
	if (value > PLY_INT8_MAX || value < PLY_INT8_MIN) return 0;
	return fprintf( ply->fp, "%d", (t_ply_int8)value ) > 0;
}

static int oascii_uint8( p_ply ply, double value )
{
	if (value > PLY_UINT8_MAX || value < 0) return 0;
	return fprintf( ply->fp, "%d", (t_ply_uint8)value ) > 0;
}

static int oascii_int16( p_ply ply, double value )
{
	if (value > PLY_INT16_MAX || value < PLY_INT16_MIN) return 0;
	return fprintf( ply->fp, "%d", (t_ply_int16)value ) > 0;
}

static int oascii_uint16( p_ply ply, double value )
{
	if (value > PLY_UINT16_MAX || value < 0) return 0;
	return fprintf( ply->fp, "%d", (t_ply_uint16)value ) > 0;
}

static int oascii_int32( p_ply ply, double value )
{
	if (value > PLY_INT32_MAX || value < PLY_INT32_MIN) return 0;
	return fprintf( ply->fp, "%d", (t_ply_int32)value ) > 0;
}

static int oascii_uint32( p_ply ply, double value )
{
	if (value > PLY_UINT32_MAX || value < 0) return 0;
	return fprintf( ply->fp, "%d", (t_ply_uint32)value ) > 0;
}

static int oascii_float32( p_ply ply, double value )
{
	if (value < -FLT_MAX || value > FLT_MAX) return 0;
	return fprintf( ply->fp, "%g", (float)value ) > 0;
}

static int oascii_float64( p_ply ply, double value )
{
	if (value < -DBL_MAX || value > DBL_MAX) return 0;
	return fprintf( ply->fp, "%g", value ) > 0;
}

static int obinary_int8( p_ply ply, double value )
{
	t_ply_int8 int8 = (t_ply_int8)value;
	if (value > PLY_INT8_MAX || value < PLY_INT8_MIN) return 0;
	return ply->odriver->ochunk( ply, &int8, sizeof( int8 ) );
}

static int obinary_uint8( p_ply ply, double value )
{
	t_ply_uint8 uint8 = (t_ply_uint8)value;
	if (value > PLY_UINT8_MAX || value < 0) return 0;
	return ply->odriver->ochunk( ply, &uint8, sizeof( uint8 ) );
}

static int obinary_int16( p_ply ply, double value )
{
	t_ply_int16 int16 = (t_ply_int16)value;
	if (value > PLY_INT16_MAX || value < PLY_INT16_MIN) return 0;
	return ply->odriver->ochunk( ply, &int16, sizeof( int16 ) );
}

static int obinary_uint16( p_ply ply, double value )
{
	t_ply_uint16 uint16 = (t_ply_uint16)value;
	if (value > PLY_UINT16_MAX || value < 0) return 0;
	return ply->odriver->ochunk( ply, &uint16, sizeof( uint16 ) );
}

static int obinary_int32( p_ply ply, double value )
{
	t_ply_int32 int32 = (t_ply_int32)value;
	if (value > PLY_INT32_MAX || value < PLY_INT32_MIN) return 0;
	return ply->odriver->ochunk( ply, &int32, sizeof( int32 ) );
}

static int obinary_uint32( p_ply ply, double value )
{
	t_ply_uint32 uint32 = (t_ply_uint32)value;
	if (value > PLY_UINT32_MAX || value < 0) return 0;
	return ply->odriver->ochunk( ply, &uint32, sizeof( uint32 ) );
}

static int obinary_float32( p_ply ply, double value )
{
	float float32 = (float)value;
	if (value > FLT_MAX || value < -FLT_MAX) return 0;
	return ply->odriver->ochunk( ply, &float32, sizeof( float32 ) );
}

static int obinary_float64( p_ply ply, double value )
{
	return ply->odriver->ochunk( ply, &value, sizeof( value ) );
}

/* ----------------------------------------------------------------------
 * Input  handlers
 * ---------------------------------------------------------------------- */
static int iascii_int8( p_ply ply, double *value )
{
	char *end;
	if (!ply_read_word( ply )) return 0;
	*value = strtol( BWORD( ply ), &end, 10 );
	if (*end || *value > PLY_INT8_MAX || *value < PLY_INT8_MIN) return 0;
	return 1;
}

static int iascii_uint8( p_ply ply, double *value )
{
	char *end;
	if (!ply_read_word( ply )) return 0;
	*value = strtol( BWORD( ply ), &end, 10 );
	if (*end || *value > PLY_UINT8_MAX || *value < 0) return 0;
	return 1;
}

static int iascii_int16( p_ply ply, double *value )
{
	char *end;
	if (!ply_read_word( ply )) return 0;
	*value = strtol( BWORD( ply ), &end, 10 );
	if (*end || *value > PLY_INT16_MAX || *value < PLY_INT16_MIN) return 0;
	return 1;
}

static int iascii_uint16( p_ply ply, double *value )
{
	char *end;
	if (!ply_read_word( ply )) return 0;
	*value = strtol( BWORD( ply ), &end, 10 );
	if (*end || *value > PLY_UINT16_MAX || *value < 0) return 0;
	return 1;
}

static int iascii_int32( p_ply ply, double *value )
{
	char *end;
	if (!ply_read_word( ply )) return 0;
	*value = strtol( BWORD( ply ), &end, 10 );
	if (*end || *value > PLY_INT32_MAX || *value < PLY_INT32_MIN) return 0;
	return 1;
}

static int iascii_uint32( p_ply ply, double *value )
{
	char *end;
	if (!ply_read_word( ply )) return 0;
	*value = strtol( BWORD( ply ), &end, 10 );
	if (*end || *value > PLY_UINT32_MAX || *value < 0) return 0;
	return 1;
}

static int iascii_float32( p_ply ply, double *value )
{
	char *end;
	if (!ply_read_word( ply )) return 0;
	*value = strtod( BWORD( ply ), &end );
	if (*end || *value < -FLT_MAX || *value > FLT_MAX) return 0;
	return 1;
}

static int iascii_float64( p_ply ply, double *value )
{
	char *end;
	if (!ply_read_word( ply )) return 0;
	*value = strtod( BWORD( ply ), &end );
	if (*end || *value < -DBL_MAX || *value > DBL_MAX) return 0;
	return 1;
}

static int ibinary_int8( p_ply ply, double *value )
{
	t_ply_int8 int8;
	if (!ply->idriver->ichunk( ply, &int8, 1 )) return 0;
	*value = int8;
	return 1;
}

static int ibinary_uint8( p_ply ply, double *value )
{
	t_ply_uint8 uint8;
	if (!ply->idriver->ichunk( ply, &uint8, 1 )) return 0;
	*value = uint8;
	return 1;
}

static int ibinary_int16( p_ply ply, double *value )
{
	t_ply_int16 int16;
	if (!ply->idriver->ichunk( ply, &int16, sizeof( int16 ) )) return 0;
	*value = int16;
	return 1;
}

static int ibinary_uint16( p_ply ply, double *value )
{
	t_ply_uint16 uint16;
	if (!ply->idriver->ichunk( ply, &uint16, sizeof( uint16 ) )) return 0;
	*value = uint16;
	return 1;
}

static int ibinary_int32( p_ply ply, double *value )
{
	t_ply_int32 int32;
	if (!ply->idriver->ichunk( ply, &int32, sizeof( int32 ) )) return 0;
	*value = int32;
	return 1;
}

static int ibinary_uint32( p_ply ply, double *value )
{
	t_ply_uint32 uint32;
	if (!ply->idriver->ichunk( ply, &uint32, sizeof( uint32 ) )) return 0;
	*value = uint32;
	return 1;
}

static int ibinary_float32( p_ply ply, double *value )
{
	float float32;
	if (!ply->idriver->ichunk( ply, &float32, sizeof( float32 ) )) return 0;
	*value = float32;
	return 1;
}

static int ibinary_float64( p_ply ply, double *value )
{
	return ply->idriver->ichunk( ply, value, sizeof( double ) );
}

/* ----------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */

namespace
{
t_ply_idriver ply_idriver_ascii = {
	{iascii_int8, iascii_uint8, iascii_int16, iascii_uint16, iascii_int32,
	 iascii_uint32, iascii_float32, iascii_float64, iascii_int8, iascii_uint8,
	 iascii_int16, iascii_uint16, iascii_int32, iascii_uint32, iascii_float32,
	 iascii_float64}, /* order matches e_ply_type enum */
	NULL,
	"ascii input" };

t_ply_idriver ply_idriver_binary = {
	{ibinary_int8, ibinary_uint8, ibinary_int16, ibinary_uint16, ibinary_int32,
	 ibinary_uint32, ibinary_float32, ibinary_float64, ibinary_int8,
	 ibinary_uint8, ibinary_int16, ibinary_uint16, ibinary_int32,
	 ibinary_uint32, ibinary_float32,
	 ibinary_float64}, /* order matches e_ply_type enum */
	ply_read_chunk,
	"binary input" };

t_ply_idriver ply_idriver_binary_reverse = {
	{ibinary_int8, ibinary_uint8, ibinary_int16, ibinary_uint16, ibinary_int32,
	 ibinary_uint32, ibinary_float32, ibinary_float64, ibinary_int8,
	 ibinary_uint8, ibinary_int16, ibinary_uint16, ibinary_int32,
	 ibinary_uint32, ibinary_float32,
	 ibinary_float64}, /* order matches e_ply_type enum */
	ply_read_chunk_reverse,
	"reverse binary input" };
};

/* ----------------------------------------------------------------------
 * Copyright (C) 2003-2011 Diego Nehab.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ---------------------------------------------------------------------- */