/* navmesh_saving.h - Copyright 2019 Utrecht University
   
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

#pragma once

#include <DetourNavMesh.h> // dtAllocNavMesh, dtNavMesh
#include <Recast.h>

#include "navmesh_common.h" // NavMeshStatus, NavMeshError
#include "tinyxml2.h"

#define NAVMESH_IO_ERROR(...) return NavMeshError(0, NavMeshStatus::IO, "ERROR NavMesh IO: ", __VA_ARGS__)
#define PF_CONFIG_FILE_ROOT_NAME "configurations"
#define PF_OMC_FILE_ROOT_NAME "OMCs"
#define PF_PMESH_FILE_ROOT_NAME "pmesh"
#define PF_DMESH_FILE_ROOT_NAME "dmesh"

namespace lighthouse2 {
	
//  +-----------------------------------------------------------------------------+
//  |  SerializeEndChild                                                          |
//  |  Adds the value as an end child with the given name.						  |
//  |  This works for int/char/short/long, uint/uchar/ushort/DWORD,				  |
//  |  float/double, bool, std::string, int2/float2/uint2, int3/float3/uint3,	  |
//  |  int4/float4/uint4/uchar4, and std::vectors containing these types.   LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type>
static void SerializeEndChild(tinyxml2::XMLNode* parent, const char* name, const type value, tinyxml2::XMLDocument& doc)
	{ ((tinyxml2::XMLElement*)parent->InsertEndChild(doc.NewElement(name)))->SetText(value); }
template <>
static void SerializeEndChild<std::string>(tinyxml2::XMLNode* parent, const char* name, const std::string value, tinyxml2::XMLDocument& doc)
	{ ((tinyxml2::XMLElement*)parent->InsertEndChild(doc.NewElement(name)))->SetText(value.c_str()); }

//  +-----------------------------------------------------------------------------+
//  |  SerializeEndChild2                                                         |
//  |  Adds the values of a double as one end child with the given name.		  |
//  |  This works for int2/float2/uint2.								    LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type>
static void SerializeEndChild2(tinyxml2::XMLNode* parent, const char* name, const type value, tinyxml2::XMLDocument& doc)
{
	tinyxml2::XMLNode* item = parent->InsertEndChild(doc.NewElement(name));
	SerializeEndChild(item, "x", value.x, doc);
	SerializeEndChild(item, "y", value.y, doc);
}
template <> static void SerializeEndChild<int2>(tinyxml2::XMLNode* parent, const char* name, const int2 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild2(parent, name, value, doc); }
template <> static void SerializeEndChild<float2>(tinyxml2::XMLNode* parent, const char* name, const float2 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild2(parent, name, value, doc); }
template <> static void SerializeEndChild<uint2>(tinyxml2::XMLNode* parent, const char* name, const uint2 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild2(parent, name, value, doc); }

//  +-----------------------------------------------------------------------------+
//  |  SerializeEndChild3                                                         |
//  |  Adds the values of a triple as one end child with the given name.		  |
//  |  This works for int3/float3/uint3.								    LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type>
static void SerializeEndChild3(tinyxml2::XMLNode* parent, const char* name, const type value, tinyxml2::XMLDocument& doc)
{
	tinyxml2::XMLNode* item = parent->InsertEndChild(doc.NewElement(name));
	SerializeEndChild(item, "x", value.x, doc);
	SerializeEndChild(item, "y", value.y, doc);
	SerializeEndChild(item, "z", value.z, doc);
}
template <> static void SerializeEndChild<int3>(tinyxml2::XMLNode* parent, const char* name, const int3 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild3(parent, name, value, doc); }
template <> static void SerializeEndChild<float3>(tinyxml2::XMLNode* parent, const char* name, const float3 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild3(parent, name, value, doc); }
template <> static void SerializeEndChild<uint3>(tinyxml2::XMLNode* parent, const char* name, const uint3 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild3(parent, name, value, doc); }

//  +-----------------------------------------------------------------------------+
//  |  SerializeEndChild4                                                         |
//  |  Adds the values of a quad as one end child with the given name.			  |
//  |  This works for int4/float4/uint4/uchar4.							    LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type>
static void SerializeEndChild4(tinyxml2::XMLNode* parent, const char* name, const type value, tinyxml2::XMLDocument& doc)
{
	tinyxml2::XMLNode* item = parent->InsertEndChild(doc.NewElement(name));
	SerializeEndChild(item, "x", value.x, doc);
	SerializeEndChild(item, "y", value.y, doc);
	SerializeEndChild(item, "z", value.z, doc);
	SerializeEndChild(item, "w", value.w, doc);
}
template <> static void SerializeEndChild<int4>(tinyxml2::XMLNode* parent, const char* name, const int4 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild4(parent, name, value, doc); }
template <> static void SerializeEndChild<float4>(tinyxml2::XMLNode* parent, const char* name, const float4 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild4(parent, name, value, doc); }
template <> static void SerializeEndChild<uint4>(tinyxml2::XMLNode* parent, const char* name, const uint4 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild4(parent, name, value, doc); }
template <> static void SerializeEndChild<uchar4>(tinyxml2::XMLNode* parent, const char* name, const uchar4 value, tinyxml2::XMLDocument& doc)
	{ SerializeEndChild4(parent, name, value, doc); }

//  +-----------------------------------------------------------------------------+
//  |  SerializeEndChild                                                          |
//  |  Adds the vector as one end child with the given name. It includes		  |
//  |  a "size" field, and an "items" field with "item" children.			LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type>
static void SerializeEndChild(tinyxml2::XMLNode* parent, const char* name, const std::vector<type> vector, tinyxml2::XMLDocument& doc)
{
	// Initialize vector container
	tinyxml2::XMLNode* list = doc.NewElement(name);
	parent->InsertEndChild(list);
	((tinyxml2::XMLElement*)list->InsertEndChild(doc.NewElement("size")))->SetText((uint)vector.size());

	// Serialize vector items
	tinyxml2::XMLNode* items = doc.NewElement("items");
	list->InsertEndChild(items);
	for (auto a : vector) SerializeEndChild<type>(items, "item", a, doc);
}

//  +-----------------------------------------------------------------------------+
//  |  DeserializeItem		                                                      |
//  |  Retrieves the value from a serialized item.						  		  |
//  |  This works for int/char/short/long, uint/uchar/ushort/DWORD,	float/double, |
//  |  bool, std::string, float3, and std::vectors containing these types.  LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type> static type DeserializeItem(tinyxml2::XMLElement* item) { return item->Value(); };
template <> static int DeserializeItem<int>(tinyxml2::XMLElement* item) { int v; item->QueryIntText(&v); return v; };
template <> static char DeserializeItem<char>(tinyxml2::XMLElement* item) { int v; item->QueryIntText(&v); return (char)v; };
template <> static short DeserializeItem<short>(tinyxml2::XMLElement* item) { int v; item->QueryIntText(&v); return (short)v; };
template <> static long DeserializeItem<long>(tinyxml2::XMLElement* item) { int v; item->QueryIntText(&v); return (long)v; };
template <> static uint DeserializeItem<uint>(tinyxml2::XMLElement* item) { uint v; item->QueryUnsignedText(&v); return v; };
template <> static uchar DeserializeItem<uchar>(tinyxml2::XMLElement* item) { uint v; item->QueryUnsignedText(&v); return (uchar)v; };
template <> static ushort DeserializeItem<ushort>(tinyxml2::XMLElement* item) { uint v; item->QueryUnsignedText(&v); return (ushort)v; };
template <> static DWORD DeserializeItem<DWORD>(tinyxml2::XMLElement* item) { uint v; item->QueryUnsignedText(&v); return (DWORD)v; };
template <> static float DeserializeItem<float>(tinyxml2::XMLElement* item) { float v; item->QueryFloatText(&v); return v; };
template <> static double DeserializeItem<double>(tinyxml2::XMLElement* item) { double v; item->QueryDoubleText(&v); return v; };
template <> static bool DeserializeItem<bool>(tinyxml2::XMLElement* item) { bool v; item->QueryBoolText(&v); return v; };
template <> static std::string DeserializeItem<std::string>(tinyxml2::XMLElement* item) { return item->FirstChild()->Value(); };

//  +-----------------------------------------------------------------------------+
//  |  DeserializeFirstChild                                                      |
//  |  Copies the first child with the given name to the destination.			  |
//  |  This works for int/char/short/long, uint/uchar/ushort/DWORD,				  |
//  |  float/double, bool, std::string, int2/float2/uint2, int3/float3/uint3,	  |
//  |  int4/float4/uint4/uchar4, and std::vectors containing these types.   LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type>
static void DeserializeFirstChild(tinyxml2::XMLElement* parent, const char* name, type& dst)
{
	tinyxml2::XMLElement* child = parent->FirstChildElement(name);
	if (child) dst = DeserializeItem<type>(child);
}

//  +-----------------------------------------------------------------------------+
//  |  DeserializeItem2		                                                      |
//  |  Retrieves the values of a double from a serialized item.					  |
//  |  This works for int2/float2/uint2.								    LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type, typename itemtype> static type DeserializeItem2(tinyxml2::XMLElement* item)
{
	type v;
	DeserializeFirstChild<itemtype>(item, "x", v.x);
	DeserializeFirstChild<itemtype>(item, "y", v.y);
	return v;
}
template <> static int2 DeserializeItem<int2>(tinyxml2::XMLElement* item)
	{ return DeserializeItem2<int2, int>(item); };
template <> static float2 DeserializeItem<float2>(tinyxml2::XMLElement* item)
	{ return DeserializeItem2<float2, float>(item); };
template <> static uint2 DeserializeItem<uint2>(tinyxml2::XMLElement* item)
	{ return DeserializeItem2<uint2, uint>(item); };

//  +-----------------------------------------------------------------------------+
//  |  DeserializeItem3		                                                      |
//  |  Retrieves the values of a triple from a serialized item.					  |
//  |  This works for int3/float3/uint3.								    LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type, typename itemtype> static type DeserializeItem3(tinyxml2::XMLElement* item)
{
	type v;
	DeserializeFirstChild<itemtype>(item, "x", v.x);
	DeserializeFirstChild<itemtype>(item, "y", v.y);
	DeserializeFirstChild<itemtype>(item, "z", v.z);
	return v;
}
template <> static int3 DeserializeItem<int3>(tinyxml2::XMLElement* item)
	{ return DeserializeItem3<int3, int>(item); };
template <> static float3 DeserializeItem<float3>(tinyxml2::XMLElement* item)
	{ return DeserializeItem3<float3, float>(item); };
template <> static uint3 DeserializeItem<uint3>(tinyxml2::XMLElement* item)
	{ return DeserializeItem3<uint3, uint>(item); };

//  +-----------------------------------------------------------------------------+
//  |  DeserializeItem4		                                                      |
//  |  Retrieves the values of a quad from a serialized item.					  |
//  |  This works for int4/float4/uint4/uchar4.							    LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type, typename itemtype> static type DeserializeItem4(tinyxml2::XMLElement* item)
{
	type v;
	DeserializeFirstChild<itemtype>(item, "x", v.x);
	DeserializeFirstChild<itemtype>(item, "y", v.y);
	DeserializeFirstChild<itemtype>(item, "z", v.z);
	DeserializeFirstChild<itemtype>(item, "w", v.w);
	return v;
}
template <> static int4 DeserializeItem<int4>(tinyxml2::XMLElement* item)
	{ return DeserializeItem4<int4, int>(item); };
template <> static float4 DeserializeItem<float4>(tinyxml2::XMLElement* item)
	{ return DeserializeItem4<float4, float>(item); };
template <> static uint4 DeserializeItem<uint4>(tinyxml2::XMLElement* item)
	{ return DeserializeItem4<uint4, uint>(item); };
template <> static uchar4 DeserializeItem<uchar4>(tinyxml2::XMLElement* item)
	{ return DeserializeItem4<uchar4, uchar>(item); };

//  +-----------------------------------------------------------------------------+
//  |  DeserializeFirstChild                                                      |
//  |  Copies the first vector child with the given name to the destination. It   |
//  |  reads a "size" field, and an "items" field with "item" children.		LH2'19|
//  +-----------------------------------------------------------------------------+
template <typename type>
static void DeserializeFirstChild(tinyxml2::XMLNode* parent, const char* name, std::vector<type>& dst)
{
	// Retrieve vector size
	int size = 0;
	if (parent->FirstChildElement(name) && parent->FirstChildElement(name)->FirstChildElement("size"))
		parent->FirstChildElement(name)->FirstChildElement("size")->QueryIntText(&size);

	// Iterate vector items
	if (size > 0 && parent->FirstChildElement(name)->FirstChildElement("items"))
	{
		dst.resize(size);
		tinyxml2::XMLElement* item = parent->FirstChildElement(name)->FirstChildElement("items")->FirstChildElement("item");
		for (int i = 0; i < size; i++)
		{
			if (item) dst[i] = DeserializeItem<type>(item);
			else break;
			item = item->NextSiblingElement("item");
		}
	}
}







//  +-----------------------------------------------------------------------------+
//  |  SerializePolyMesh                                                          |
//  |  Saves an rcPolyMesh.												    LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus SerializePolyMesh(std::string filename, const rcPolyMesh* pmesh)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLNode* root = doc.NewElement(PF_PMESH_FILE_ROOT_NAME);
	doc.InsertFirstChild(root);
	
	SerializeEndChild(root, "verts", std::vector<ushort>(pmesh->verts, pmesh->verts + pmesh->nverts * 3), doc);
	SerializeEndChild(root, "polys", std::vector<ushort>(pmesh->polys, pmesh->polys + pmesh->maxpolys * 2 * pmesh->nvp), doc);
	SerializeEndChild(root, "regs",  std::vector<ushort>(pmesh->regs,  pmesh->regs  + pmesh->maxpolys), doc);
	SerializeEndChild(root, "flags", std::vector<ushort>(pmesh->flags, pmesh->flags + pmesh->maxpolys), doc);
	SerializeEndChild(root, "areas", std::vector<uchar>(pmesh->areas,  pmesh->areas + pmesh->maxpolys), doc);
	SerializeEndChild(root, "nverts", pmesh->nverts, doc);
	SerializeEndChild(root, "npolys", pmesh->npolys, doc);
	SerializeEndChild(root, "maxpolys", pmesh->maxpolys, doc);
	SerializeEndChild(root, "nvp", pmesh->nvp, doc);
	SerializeEndChild(root, "bmin", *((float3*)pmesh->bmin), doc);
	SerializeEndChild(root, "bmax", *((float3*)pmesh->bmax), doc);
	SerializeEndChild(root, "cs", pmesh->cs, doc);
	SerializeEndChild(root, "ch", pmesh->ch, doc);
	SerializeEndChild(root, "borderSize", pmesh->borderSize, doc);
	SerializeEndChild(root, "maxEdgeError", pmesh->maxEdgeError, doc);

	tinyxml2::XMLError result = doc.SaveFile(filename.c_str());
	if (result != tinyxml2::XML_SUCCESS)
		NAVMESH_IO_ERROR("PMesh file '%s' could not be saved\n", filename.c_str());
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  DeserializePolyMesh                                                        |
//  |  Loads an rcPolyMesh.												    LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus DeserializePolyMesh(std::string filename, rcPolyMesh*& pmesh)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLError result = doc.LoadFile(filename.c_str());
	if (result != tinyxml2::XML_SUCCESS)
		NAVMESH_IO_ERROR("PMesh file '%s' could not be opened\n", filename.c_str());
	tinyxml2::XMLElement* root = doc.FirstChildElement(PF_PMESH_FILE_ROOT_NAME);
	if (root == nullptr)
		NAVMESH_IO_ERROR("tinyXML2 errored while loading PMesh file '%s'\n", filename.c_str());
	if (!pmesh) pmesh = new rcPolyMesh();

	std::vector<unsigned short> verts;
	std::vector<unsigned short> polys;
	std::vector<unsigned short> regs;
	std::vector<unsigned short> flags;
	std::vector<unsigned char> areas;
	DeserializeFirstChild<ushort>(root, "verts", verts);
	DeserializeFirstChild<ushort>(root, "polys", polys);
	DeserializeFirstChild<ushort>(root, "regs", regs);
	DeserializeFirstChild<ushort>(root, "flags", flags);
	DeserializeFirstChild<uchar>(root, "areas", areas);
	pmesh->verts = new ushort[verts.size()];
	pmesh->polys = new ushort[polys.size()];
	pmesh->regs = new ushort[regs.size()];
	pmesh->flags = new ushort[flags.size()];
	pmesh->areas = new uchar[areas.size()];
	memcpy(pmesh->verts, verts.data(), verts.size() * sizeof(ushort));
	memcpy(pmesh->polys, polys.data(), polys.size() * sizeof(ushort));
	memcpy(pmesh->regs, regs.data(), regs.size() * sizeof(ushort));
	memcpy(pmesh->flags, flags.data(), flags.size() * sizeof(ushort));
	memcpy(pmesh->areas, areas.data(), areas.size() * sizeof(uchar));
	DeserializeFirstChild<int>(root, "nverts", pmesh->nverts);
	DeserializeFirstChild<int>(root, "npolys", pmesh->npolys);
	DeserializeFirstChild<int>(root, "maxpolys", pmesh->maxpolys);
	DeserializeFirstChild<int>(root, "nvp", pmesh->nvp);
	DeserializeFirstChild<float3>(root, "bmin", *((float3*)pmesh->bmin));
	DeserializeFirstChild<float3>(root, "bmax", *((float3*)pmesh->bmax));
	DeserializeFirstChild<float>(root, "cs", pmesh->cs);
	DeserializeFirstChild<float>(root, "ch", pmesh->ch);
	DeserializeFirstChild<int>(root, "borderSize", pmesh->borderSize);
	DeserializeFirstChild<float>(root, "maxEdgeError", pmesh->maxEdgeError);

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  SerializeDetailMesh                                                        |
//  |  Saves an rcPolyMeshDetail.										    LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus SerializeDetailMesh(std::string filename, const rcPolyMeshDetail* dmesh)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLNode* root = doc.NewElement(PF_DMESH_FILE_ROOT_NAME);
	doc.InsertFirstChild(root);

	// converting to uint4/float3/uchar4 to improve readability of save files
	std::vector<uint4> meshes((uint4*)dmesh->meshes, (uint4*)dmesh->meshes + dmesh->nmeshes);
	std::vector<float3> verts((float3*)dmesh->verts, (float3*)dmesh->verts + dmesh->nverts);
	std::vector<uchar4> tris((uchar4*)dmesh->tris, (uchar4*)dmesh->tris + dmesh->ntris);
	SerializeEndChild(root, "meshes", meshes, doc);
	SerializeEndChild(root, "verts", verts, doc);
	SerializeEndChild(root, "tris", tris, doc);
	SerializeEndChild(root, "nmeshes", dmesh->nmeshes, doc);
	SerializeEndChild(root, "nverts", dmesh->nverts, doc);
	SerializeEndChild(root, "ntris", dmesh->ntris, doc);

	tinyxml2::XMLError result = doc.SaveFile(filename.c_str());
	if (result != tinyxml2::XML_SUCCESS)
		NAVMESH_IO_ERROR("DMesh file '%s' could not be saved\n", filename.c_str());
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  DeserializeDetailMesh                                                      |
//  |  Loads an rcPolyMeshDetail.										    LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus DeserializeDetailMesh(std::string filename, rcPolyMeshDetail*& dmesh)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLError result = doc.LoadFile(filename.c_str());
	if (result != tinyxml2::XML_SUCCESS)
		NAVMESH_IO_ERROR("DMesh file '%s' could not be opened\n", filename.c_str());
	tinyxml2::XMLElement* root = doc.FirstChildElement(PF_DMESH_FILE_ROOT_NAME);
	if (root == nullptr)
		NAVMESH_IO_ERROR("tinyXML2 errored while loading DMesh file '%s'\n", filename.c_str());
	if (!dmesh) dmesh = new rcPolyMeshDetail();

	std::vector<uint4> meshes;
	std::vector<float3> verts;
	std::vector<uchar4> tris;
	DeserializeFirstChild<uint4>(root, "meshes", meshes);
	DeserializeFirstChild<float3>(root, "verts", verts);
	DeserializeFirstChild<uchar4>(root, "tris", tris);
	dmesh->meshes = new uint[meshes.size() * 4];
	dmesh->verts = new float[verts.size() * 3];
	dmesh->tris = new uchar[tris.size() * 4];

	// Converting back from uint4/float3/uchar4 to uint4/float/uchar
	memcpy(dmesh->meshes, meshes.data(), meshes.size() * sizeof(uint4));
	memcpy(dmesh->verts, verts.data(), verts.size() * sizeof(float3));
	memcpy(dmesh->tris, tris.data(), tris.size() * sizeof(uchar4));
	DeserializeFirstChild<int>(root, "nmeshes", dmesh->nmeshes);
	DeserializeFirstChild<int>(root, "nverts", dmesh->nverts);
	DeserializeFirstChild<int>(root, "ntris", dmesh->ntris);

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  SerializeOffMeshConnections                                                |
//  |  Saves the builder's off-mesh connections.						    LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus SerializeOffMeshConnections(
	std::string filename,
	std::vector<float3> offMeshVerts,
	std::vector<float> offMeshRadii,
	std::vector<unsigned short> offMeshFlags,
	std::vector<unsigned char> offMeshAreas,
	std::vector<unsigned int> offMeshUserIDs,
	std::vector<unsigned char> offMeshDirection
)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLNode* root = doc.NewElement(PF_OMC_FILE_ROOT_NAME);
	doc.InsertFirstChild(root);

	SerializeEndChild(root, "verts", offMeshVerts, doc);
	SerializeEndChild(root, "radii", offMeshRadii, doc);
	SerializeEndChild(root, "flags", offMeshFlags, doc);
	SerializeEndChild(root, "areas", offMeshAreas, doc);
	SerializeEndChild(root, "userIDs", offMeshUserIDs, doc);
	SerializeEndChild(root, "directions", offMeshDirection, doc);

	tinyxml2::XMLError result = doc.SaveFile(filename.c_str());
	if (result != tinyxml2::XML_SUCCESS)
		NAVMESH_IO_ERROR("OMC file '%s' could not be saved\n", filename.c_str());
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  DeserializeOffMeshConnections                                              |
//  |  Saves the builder's off-mesh connections.						    LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus DeserializeOffMeshConnections(
	std::string filename,
	std::vector<float3>& offMeshVerts,
	std::vector<float>& offMeshRadii,
	std::vector<unsigned short>& offMeshFlags,
	std::vector<unsigned char>& offMeshAreas,
	std::vector<unsigned int>& offMeshUserIDs,
	std::vector<unsigned char>& offMeshDirection
)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLError result = doc.LoadFile(filename.c_str());
	if (result != tinyxml2::XML_SUCCESS)
		NAVMESH_IO_ERROR("OMC file '%s' could not be opened\n", filename.c_str());
	tinyxml2::XMLElement* root = doc.FirstChildElement(PF_OMC_FILE_ROOT_NAME);
	if (root == nullptr)
		NAVMESH_IO_ERROR("tinyXML2 errored while loading OMC file '%s'\n", filename.c_str());

	DeserializeFirstChild<float3>(root, "verts", offMeshVerts);
	DeserializeFirstChild<float>(root, "radii", offMeshRadii);
	DeserializeFirstChild<ushort>(root, "flags", offMeshFlags);
	DeserializeFirstChild<uchar>(root, "areas", offMeshAreas);
	DeserializeFirstChild<uint>(root, "userIDs", offMeshUserIDs);
	DeserializeFirstChild<uchar>(root, "directions", offMeshDirection);

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  SerializeConfigurations		                                              |
//  |  Saves the builder's configurations.								    LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus SerializeConfigurations(std::string filename, const NavMeshConfig& config)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLNode* root = doc.NewElement(PF_CONFIG_FILE_ROOT_NAME);
	doc.InsertFirstChild(root);

	SerializeEndChild(root, "width", config.m_width, doc);
	SerializeEndChild(root, "height", config.m_height, doc);
	SerializeEndChild(root, "tileSize", config.m_tileSize, doc);
	SerializeEndChild(root, "borderSize", config.m_borderSize, doc);

	SerializeEndChild(root, "cs", config.m_cs, doc);
	SerializeEndChild(root, "ch", config.m_ch, doc);
	SerializeEndChild(root, "bmin", config.m_bmin, doc);
	SerializeEndChild(root, "bmax", config.m_bmax, doc);

	SerializeEndChild(root, "walkableSlopeAngle", config.m_walkableSlopeAngle, doc);
	SerializeEndChild(root, "walkableHeight", config.m_walkableHeight, doc);
	SerializeEndChild(root, "walkableClimb", config.m_walkableClimb, doc);
	SerializeEndChild(root, "walkableRadius", config.m_walkableRadius, doc);

	SerializeEndChild(root, "maxEdgeLen", config.m_maxEdgeLen, doc);
	SerializeEndChild(root, "maxSimplificationError", config.m_maxSimplificationError, doc);
	SerializeEndChild(root, "minRegionArea", config.m_minRegionArea, doc);
	SerializeEndChild(root, "mergeRegionArea", config.m_mergeRegionArea, doc);
	SerializeEndChild(root, "maxVertsPerPoly", config.m_maxVertsPerPoly, doc);
	SerializeEndChild(root, "detailSampleDist", config.m_detailSampleDist, doc);
	SerializeEndChild(root, "detailSampleMaxError", config.m_detailSampleMaxError, doc);

	SerializeEndChild(root, "partitionType", (uchar)config.m_partitionType, doc);
	SerializeEndChild(root, "keepInterResults", config.m_keepInterResults, doc);
	SerializeEndChild(root, "filterLowHangingObstacles", config.m_filterLowHangingObstacles, doc);
	SerializeEndChild(root, "filterLedgeSpans", config.m_filterLedgeSpans, doc);
	SerializeEndChild(root, "filterWalkableLowHeightSpans", config.m_filterWalkableLowHeightSpans, doc);

	SerializeEndChild(root, "printBuildStats", config.m_printBuildStats, doc);
	SerializeEndChild(root, "ID", config.m_id, doc);

	SerializeEndChild(root, "arealabels", config.m_areas.labels, doc);
	SerializeEndChild(root, "areaDefaultCosts", config.m_areas.defaultCosts, doc);
	SerializeEndChild(root, "flaglabels", config.m_flags.labels, doc);

	tinyxml2::XMLError result = doc.SaveFile(filename.c_str());
	if (result != tinyxml2::XML_SUCCESS)
		NAVMESH_IO_ERROR("Config file '%s' could not be saved\n", filename.c_str());
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  DeserializeConfigurations	                                              |
//  |  Loads the builder's configurations.								    LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus DeserializeConfigurations(std::string filename, NavMeshConfig& config)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLError result = doc.LoadFile(filename.c_str());
	if (result != tinyxml2::XML_SUCCESS)
		NAVMESH_IO_ERROR("Config file '%s' could not be opened\n", filename.c_str());
	tinyxml2::XMLElement* root = doc.FirstChildElement(PF_CONFIG_FILE_ROOT_NAME);
	if (root == nullptr)
		NAVMESH_IO_ERROR("tinyXML2 errored while loading config file '%s'\n", filename.c_str());

	DeserializeFirstChild<int>(root, "width", config.m_width);
	DeserializeFirstChild<int>(root, "height", config.m_height);
	DeserializeFirstChild<int>(root, "tileSize", config.m_tileSize);
	DeserializeFirstChild<int>(root, "borderSize", config.m_borderSize);

	DeserializeFirstChild<float>(root, "cs", config.m_cs);
	DeserializeFirstChild<float>(root, "ch", config.m_ch);
	DeserializeFirstChild<float3>(root, "bmin", config.m_bmin);
	DeserializeFirstChild<float3>(root, "bmax", config.m_bmax);

	DeserializeFirstChild<float>(root, "walkableSlopeAngle", config.m_walkableSlopeAngle);
	DeserializeFirstChild<int>(root, "walkableHeight", config.m_walkableHeight);
	DeserializeFirstChild<int>(root, "walkableClimb", config.m_walkableClimb);
	DeserializeFirstChild<int>(root, "walkableRadius", config.m_walkableRadius);

	DeserializeFirstChild<int>(root, "maxEdgeLen", config.m_maxEdgeLen);
	DeserializeFirstChild<float>(root, "maxSimplificationError", config.m_maxSimplificationError);
	DeserializeFirstChild<int>(root, "minRegionArea", config.m_minRegionArea);
	DeserializeFirstChild<int>(root, "mergeRegionArea", config.m_mergeRegionArea);
	DeserializeFirstChild<int>(root, "maxVertsPerPoly", config.m_maxVertsPerPoly);
	DeserializeFirstChild<float>(root, "detailSampleDist", config.m_detailSampleDist);
	DeserializeFirstChild<float>(root, "detailSampleMaxError", config.m_detailSampleMaxError);

	DeserializeFirstChild<uchar>(root, "partitionType", *((uchar*)&config.m_partitionType));
	DeserializeFirstChild<bool>(root, "keepInterResults", config.m_keepInterResults);
	DeserializeFirstChild<bool>(root, "filterLowHangingObstacles", config.m_filterLowHangingObstacles);
	DeserializeFirstChild<bool>(root, "filterLedgeSpans", config.m_filterLedgeSpans);
	DeserializeFirstChild<bool>(root, "filterWalkableLowHeightSpans", config.m_filterWalkableLowHeightSpans);

	DeserializeFirstChild<bool>(root, "printBuildStats", config.m_printBuildStats);
	DeserializeFirstChild<std::string>(root, "ID", config.m_id);

	DeserializeFirstChild<std::string>(root, "arealabels", config.m_areas.labels);
	DeserializeFirstChild<float>(root, "areaDefaultCosts", config.m_areas.defaultCosts);
	DeserializeFirstChild<std::string>(root, "flaglabels", config.m_flags.labels);

	return NavMeshStatus::SUCCESS;
}



// Definitions required for NavMesh serialization
static const int NAVMESHSET_MAGIC = 'M' << 24 | 'S' << 16 | 'E' << 8 | 'T'; //'MSET';
static const int NAVMESHSET_VERSION = 1;
struct NavMeshSetHeader { int magic, version, numTiles; dtNavMeshParams params; };
struct NavMeshTileHeader { dtTileRef tileRef; int dataSize; };

//  +-----------------------------------------------------------------------------+
//  |  Serialize                                                                  |
//  |  Writes the navmesh to storage for future use.                        LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus SerializeNavMesh(const char* dir, const char* ID, const dtNavMesh* navMesh)
{

	// Opening navmesh file for writing
	std::string filename = std::string(dir) + ID + PF_NAVMESH_FILE_EXTENTION;
	if (!navMesh) NAVMESH_IO_ERROR("Can't serialize '%s', dtNavMesh is nullptr\n", ID);
	FILE* fp;
	fopen_s(&fp, filename.c_str(), "wb");
	if (!fp) NAVMESH_IO_ERROR("NavMesh file '%s' can't be opened\n", filename.c_str());

	// Store header.
	NavMeshSetHeader header;
	header.magic = NAVMESHSET_MAGIC;
	header.version = NAVMESHSET_VERSION;
	header.numTiles = 0;
	for (int i = 0; i < navMesh->getMaxTiles(); ++i)
	{
		const dtMeshTile* tile = navMesh->getTile(i);
		if (!tile || !tile->header || !tile->dataSize) continue;
		header.numTiles++;
	}
	memcpy(&header.params, navMesh->getParams(), sizeof(dtNavMeshParams));
	fwrite(&header, sizeof(NavMeshSetHeader), 1, fp);

	// Store tiles.
	for (int i = 0; i < navMesh->getMaxTiles(); ++i)
	{
		const dtMeshTile* tile = navMesh->getTile(i);
		if (!tile || !tile->header || !tile->dataSize) continue;

		NavMeshTileHeader tileHeader;
		tileHeader.tileRef = navMesh->getTileRef(tile);
		tileHeader.dataSize = tile->dataSize;
		fwrite(&tileHeader, sizeof(tileHeader), 1, fp);

		fwrite(tile->data, tile->dataSize, 1, fp);
	}
	fclose(fp);

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  DeserializeNavMesh                                                         |
//  |  Loads a serialized NavMesh from storage and checks for errors.       LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus DeserializeNavMesh(const char* dir, const char* ID, dtNavMesh*& navmesh)
{
	// Opening file
	std::string filename = std::string(dir) + ID + PF_NAVMESH_FILE_EXTENTION;
	if (!FileExists(filename.c_str()))
		NAVMESH_IO_ERROR("NavMesh file '%s' does not exist\n", filename.c_str());
	FILE* fp;
	fopen_s(&fp, filename.c_str(), "rb");
	if (!fp)
		NAVMESH_IO_ERROR("NavMesh file '%s' could not be opened\n", filename.c_str());

	// Reading header
	NavMeshSetHeader header;
	size_t readLen = fread(&header, sizeof(NavMeshSetHeader), 1, fp);
	if (readLen != 1)
	{
		fclose(fp);
		NAVMESH_IO_ERROR("NavMesh file '%s' is corrupted\n", filename.c_str());
	}
	if (header.magic != NAVMESHSET_MAGIC)
	{
		fclose(fp);
		NAVMESH_IO_ERROR("NavMesh file '%s' is corrupted\n", filename.c_str());
	}
	if (header.version != NAVMESHSET_VERSION)
	{
		fclose(fp);
		NAVMESH_IO_ERROR("NavMesh file '%s' has the wrong navmesh set version\n", filename.c_str());
	}

	// Initializing navmesh with header info
	navmesh = dtAllocNavMesh();
	if (!navmesh)
	{
		fclose(fp);
		NAVMESH_IO_ERROR("NavMesh for '%s' could not be allocated during loading\n", ID);
	}
	dtStatus status = navmesh->init(&header.params);
	if (dtStatusFailed(status))
	{
		fclose(fp);
		dtFreeNavMesh(navmesh);
		NAVMESH_IO_ERROR("NavMesh for '%s' failed to initialize during loading\n", ID);
	}

	// Reading tiles
	for (int i = 0; i < header.numTiles; ++i)
	{
		// Reading tile header
		NavMeshTileHeader tileHeader;
		readLen = fread(&tileHeader, sizeof(tileHeader), 1, fp);
		if (readLen != 1)
		{
			fclose(fp);
			dtFreeNavMesh(navmesh);
			NAVMESH_IO_ERROR("NavMesh file '%s' is corrupted\n", filename.c_str());
		}
		if (!tileHeader.tileRef || !tileHeader.dataSize) break;

		// Reading tile data
		unsigned char* data = (unsigned char*)dtAlloc(tileHeader.dataSize, DT_ALLOC_PERM);
		if (!data) break;
		memset(data, 0, tileHeader.dataSize);
		readLen = fread(data, tileHeader.dataSize, 1, fp);
		if (readLen != 1)
		{
			dtFree(data);
			dtFreeNavMesh(navmesh);
			fclose(fp);
			NAVMESH_IO_ERROR("NavMesh file '%s' is corrupted\n", filename.c_str());
		}
		navmesh->addTile(data, tileHeader.dataSize, DT_TILE_FREE_DATA, tileHeader.tileRef, 0);
	}
	fclose(fp);

	return NavMeshStatus::SUCCESS;
}

} // namespace lighthouse2

// EOF