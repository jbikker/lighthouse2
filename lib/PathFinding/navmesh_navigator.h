/* navmesh_navigator.h - Copyright 2019 Utrecht University

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

#include <vector>

#include "DetourNavMeshQuery.h" // dtNavMesh, dtNavMeshQuery, dtQueryFilter

#include "system.h"			// float3
#include "navmesh_common.h"	// NavMeshError, NavMeshStatus, NavMeshAreaMapping, NavMeshFlagMapping, DETOUR_MAX_NAVMESH_NODES

namespace lighthouse2 {

NavMeshStatus SerializeNavMesh(const char* dir, const char* ID, const dtNavMesh* navmesh);
NavMeshStatus DeserializeNavMesh(const char* dir, const char* ID, dtNavMesh*& navmesh);

static dtQueryFilter DefaultFilter()
{
	dtQueryFilter filter;
	filter.setIncludeFlags(USHRT_MAX);
	filter.setExcludeFlags(0);
	for (int i = 0; i < NavMeshAreaMapping::maxAreas; i++) filter.setAreaCost(i, 1.0f);
	return filter;
}
static const dtQueryFilter s_filter = DefaultFilter(); // default empty filter, includes all flags

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator                                                           |
//  |  A wrapper class for the Detour pathfinding functionality.            LH2'19|
//  +-----------------------------------------------------------------------------+
class NavMeshNavigator
{
public:

	// constructor/destructor
	NavMeshNavigator(const char* dir, const char* ID, NavMeshStatus* errorCode=0) : m_ID(ID)
	{
		NavMeshStatus status = Load(dir, ID);
		if (errorCode) *errorCode = status;
	};
	NavMeshNavigator(dtNavMesh& navmesh, const char* ID="DEFAULT_ID", NavMeshStatus* errorCode=0) : m_ID(ID), m_navmesh(&navmesh)
	{
		NavMeshStatus status = CreateNavMeshQuery();
		if (errorCode) *errorCode = status;
		m_owner = false;
	}
	~NavMeshNavigator() { Clean(); };

	//  +-----------------------------------------------------------------------------+
	//  |  NavMeshNavigator::Load                                                     |
	//  |  Loads a navmesh from storage and initializes the query.              LH2'19|
	//  +-----------------------------------------------------------------------------+
	NavMeshStatus Load(const char* dir, const char* ID)
	{
		if (m_navmesh) Clean();
		std::string configfile = std::string(dir) + ID + PF_NAVMESH_CONFIG_FILE_EXTENTION;
		NavMeshConfig config;
		NavMeshStatus status = config.Load(configfile.c_str());
		if (status.Failed()) return status;
		m_flags = config.m_flags;
		m_areas = config.m_areas;
		status = DeserializeNavMesh(dir, ID, m_navmesh);
		if (status.Failed()) return status;
		m_owner = true;
		status = CreateNavMeshQuery();
		return status;
	}

	void SetFlagAndAreaMappings(NavMeshFlagMapping flags, NavMeshAreaMapping areas) { m_flags = flags; m_areas = areas; };

	struct PathNode { float3 pos; const dtPoly* poly; }; // dtPoly* is nullptr if not on a poly

	NavMeshStatus FindNearestPoly(float3 pos, dtPolyRef& polyID, float3& polyPos) const;
	NavMeshStatus FindClosestPointOnPoly(dtPolyRef polyID, float3 pos, float3& nearestPoint, bool* posOverPoly=0) const;
	NavMeshStatus FindPathConstSize(float3 start, float3 end, PathNode* path, int& count, bool& reachable, int maxCount=64, const dtQueryFilter* filter=&s_filter) const;
	NavMeshStatus FindPathConstSize_Legacy(float3 start, float3 end, PathNode* path, int& count, bool& reachable, int maxCount = 64, const dtQueryFilter* filter=&s_filter) const;
	NavMeshStatus FindPath(float3 start, float3 end, std::vector<PathNode>& path, bool& reachable, int maxCount=64) const;
	void Clean();

	dtQueryFilter GetFilter(std::vector<std::string> includes, std::vector<std::string> excludes) const;
	inline const dtNavMesh* GetDetourMesh() const { return m_navmesh; };
	inline const dtPoly* GetPoly(dtPolyRef ref) const;
	inline const char* GetID() const { return m_ID.c_str(); };

	void SetPolyFlags(dtPolyRef poly, unsigned short flags) { m_navmesh->setPolyFlags(poly, flags); };
	void SetAreaType(dtPolyRef poly, unsigned char area) { m_navmesh->setPolyArea(poly, area); };

protected:
	std::string m_ID;			 // A unique string identifier
	bool m_owner;				 // Whether this is the owner of the dtNavMesh
	dtNavMesh* m_navmesh = 0;    // The navmesh data
	dtNavMeshQuery* m_query = 0; // Detour object handling pathfinding queries
	NavMeshFlagMapping m_flags;  // Maps polygon flags to labels
	NavMeshAreaMapping m_areas;  // Maps polygon area types to labels
	const float m_polyFindExtention[3] = { 5.0f, 5.0f, 5.0f }; // Half the search area for FindNearestPoly calls

	int NavMeshNavigator::CreateNavMeshQuery();
};

} // namespace Lighthouse2

// EOF