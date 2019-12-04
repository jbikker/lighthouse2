/* navmesh_navigator.cpp - Copyright 2019 Utrecht University

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

#include <stdio.h> // fopen_s, sprintf_s, fclose, fwrite

#include "rendersystem.h" // FileExists
#include "navmesh_io.h"   // DeserializeConfigurations

#include "navmesh_navigator.h"

namespace lighthouse2 {

#define DETOUR_ERROR(X, ...) return NavMeshError(0, X, "ERROR NavMeshNavigator: ", __VA_ARGS__)
#define POLYPATH_SIZE 128 // size of the intermediate non-smoothed path of dtPolyRefs in 'FindPathConstSize'

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::FindPathConstSize                                        |
//  |  Wrapper for the Detour findPath function. Finds a path of PathNodes.       |
//  |  *path* and *count* are the main output (preallocated).			   		  |
//  |  *reachable* specifies whether the end poly matches the last path poly.     |
//  |  *maxCount* specifies the maximum path length in nodes.               LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus NavMeshNavigator::FindPathConstSize(float3 start, float3 end, PathNode* path, int& count, bool& reachable, int maxCount, const dtQueryFilter* filter) const
{
	if (!path)
		DETOUR_ERROR(NavMeshStatus::DT | NavMeshStatus::INPUT, "Pathfinding failed: *path* is a nullpointer\n");

	// Resolve positions into navmesh polygons
	dtPolyRef startRef, endRef;
	float3 firstPos, endPos; // tmp
	NavMeshStatus status = NavMeshStatus::SUCCESS;
	status = FindNearestPoly(start, startRef, firstPos);
	if (status.Failed()) return status;
	status = FindNearestPoly(end, endRef, endPos);
	if (status.Failed()) return status;

	// When start & end are on the same poly
	if (startRef == endRef)
	{
		path[0] = PathNode{ start, startRef };
		path[1] = PathNode{ end, endRef };
		count = 2;
		reachable = true;
		return NavMeshStatus::SUCCESS;
	}

	// Calculate path
	dtPolyRef* polyPath = (dtPolyRef*)malloc(sizeof(dtPolyRef)*POLYPATH_SIZE);
	dtStatus err = m_query->findPath(startRef, endRef, (float*)&start, (float*)&end, filter, polyPath, &count, POLYPATH_SIZE);
	if (dtStatusFailed(err))
	{
		free(polyPath);
		DETOUR_ERROR(NavMeshStatus::DT, "Couldn't find a path from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)\n",
			start.x, start.y, start.z, end.x, end.y, end.z);
	}
	reachable = (polyPath[count - 1] == endRef);

	// String pulling
	float3* straightPath = (float3*)malloc(sizeof(float3) * maxCount);
	dtPolyRef* spPolys = (dtPolyRef*)malloc(sizeof(dtPolyRef) * maxCount);
	unsigned char* spFlags = (unsigned char*)malloc(sizeof(unsigned char) * maxCount);
	err = m_query->findStraightPath((float*)&start, (float*)&end, polyPath, count, (float*)straightPath, spFlags, spPolys, &count, maxCount);
	if (dtStatusFailed(err))
	{
		free(polyPath);
		free(straightPath);
		free(spPolys);
		free(spFlags);
		DETOUR_ERROR(NavMeshStatus::DT, "Couldn't find a straight path from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)\n",
			start.x, start.y, start.z, end.x, end.y, end.z);
	}

	// Converting to PathNodes
	for (int i = 0; i < count; i++)
	{
		path[i] = PathNode{ straightPath[i], spPolys[i] };
	}

	free(polyPath);
	free(straightPath);
	free(spPolys);
	free(spFlags);

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::FindPathConstSize_Legacy                                 |
//  |  Wrapper for the Detour findPath function. Finds a path of PathNodes.       |
//  |  *path* and *count* are the output (both preallocated).		     		  |
//  |  *reachable* specifies whether the end poly matches the last path poly.     |
//  |  *maxCount* specifies the maximum path length in nodes.                     |
//  |																			  |
//  |  NOTE: This implementation returns a non-smoothed path.				LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus NavMeshNavigator::FindPathConstSize_Legacy(float3 start, float3 end, PathNode* path, int& count, bool& reachable, int maxCount, const dtQueryFilter* filter) const
{
	if (!path) DETOUR_ERROR(NavMeshStatus::DT | NavMeshStatus::INPUT, "Pathfinding failed: *path* is a nullpointer\n");

	// Resolve positions into navmesh polygons
	dtPolyRef startRef, endRef;
	float3 firstPos, endPos; // first/last pos on poly
	NavMeshStatus status = NavMeshStatus::SUCCESS;
	status = FindNearestPoly(start, startRef, firstPos);
	if (status.Failed()) return status;
	status = FindNearestPoly(end, endRef, endPos);
	if (status.Failed()) return status;

	// Add the start pos
	maxCount--;
	status = FindClosestPointOnPoly(startRef, start, firstPos);
	if (status.Failed()) return status;
	path[0] = PathNode{ firstPos, startRef };

	// When start & end are on the same poly
	if (startRef == endRef)
	{
		status = FindClosestPointOnPoly(endRef, end, endPos);
		if (status.Failed()) return status;
		path[1] = PathNode{ endPos, endRef };
		count = 2;
		reachable = true;
		return NavMeshStatus::SUCCESS;
	}

	// Calculate path
	dtPolyRef* polyPath = (dtPolyRef*)malloc(sizeof(dtPolyRef)*maxCount);
	dtStatus err = m_query->findPath(startRef, endRef, (float*)&start, (float*)&end, filter, polyPath, &count, maxCount);
	if (dtStatusFailed(err))
	{
		free(polyPath);
		DETOUR_ERROR(NavMeshStatus::DT, "Couldn't find a path from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)\n",
			start.x, start.y, start.z, end.x, end.y, end.z);
	}

	// Converting to PathNodes
	float3 iterPos = firstPos;
	for (int i = 0; i < count; i++)
	{
		status = FindClosestPointOnPoly(polyPath[i], iterPos, iterPos);
		if (status.Failed()) { free(polyPath); return status; }
		path[i + 1] = PathNode{ iterPos, polyPath[i] };
	}

	// Finding the closest valid point to the target
	bool pathComplete = (count < maxCount); // means there's room for endPos
	if (pathComplete && (endRef == polyPath[count - 1])) // complete & reachable
	{
		status = FindClosestPointOnPoly(endRef, end, endPos);
		if (status.Failed()) { free(polyPath); return status; }
		path[count + 1] = PathNode{ endPos, endRef };
		count++;
		reachable = true;
	}
	else if (pathComplete) // path ended, poly unreachable
	{
		status = FindClosestPointOnPoly(polyPath[count - 1], end, endPos);
		if (status.Failed()) { free(polyPath); return status; }
		path[count + 1] = PathNode{ endPos, polyPath[count - 1] };
		count++;
		reachable = false;
	}
	else // path incomplete, reachability possible
	{
		reachable = true;
	}
	free(polyPath);

	count++; // to include firstPos

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::FindPath                                                 |
//  |  Finds the shortest path from start to end.                                 |
//  |  *path* is std::vector of world positions to be filled.                     |
//  |  *reachable* indicates if the target seems reachable (true if unknown).     |
//  |  *maxCount* specifies the maximum number of path nodes.               LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus NavMeshNavigator::FindPath(float3 start, float3 end, std::vector<PathNode>& path, bool& reachable, int maxCount) const
{
	path.resize(maxCount);
	int pathCount;
	NavMeshStatus status = FindPathConstSize(start, end, path.data(), pathCount, reachable, maxCount);
	if (status.Failed()) return status;
	path.resize(pathCount);
	path.shrink_to_fit();
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::FindNearestPointOnPoly                                   |
//  |  Finds the nearest pos on the specified *polyID* from the given position.   |
//  |  *closest* is the output.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus NavMeshNavigator::FindClosestPointOnPoly(dtPolyRef polyID, float3 pos, float3& closest, bool* posOverPoly) const
{
	dtStatus err = m_query->closestPointOnPoly(polyID, (float*)&pos, (float*)&closest, posOverPoly);
	if (dtStatusFailed(err))
		DETOUR_ERROR(NavMeshStatus::DT, "Closest point on poly '%i' to (%.2f, %.2f, %.2f) could not be found\n",
			polyID, pos.x, pos.y, pos.z);
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::FindNearestPoly                                          |
//  |  Finds the polygon closest to the specified position. *polyID* and          |
//  |  *polyPos* are the ouput. The position of the polygon is optional.    LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus NavMeshNavigator::FindNearestPoly(float3 pos, dtPolyRef& polyID, float3& polyPos) const
{
	dtStatus status = m_query->findNearestPoly((float*)&pos, m_polyFindExtention, &s_filter, &polyID, (float*)&polyPos);
	if (dtStatusFailed(status))
		DETOUR_ERROR(NavMeshStatus::DT, "Couldn't find the nearest poly to (%.2f, %.2f, %.2f)\n", pos.x, pos.y, pos.z);
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::Clean                                                    |
//  |  Frees memory and restores default values.                            LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshNavigator::Clean()
{
	if (m_navmesh && m_owner) dtFreeNavMesh(m_navmesh);
	m_navmesh = 0;
	dtFreeNavMeshQuery(m_query);
	if (m_query) m_navmesh = 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::GetFilter                                                |
//  |  Creates a navigation filter using the given include/exclude labels.        |
//  |  Excluded flags override included flags, default is excluded.               |
//  |  Passing two empty vectors will include all flags.                    LH2'19|
//  +-----------------------------------------------------------------------------+
dtQueryFilter NavMeshNavigator::GetFilter(std::vector<std::string> includes, std::vector<std::string> excludes) const
{
	unsigned short incl = 0, excl = 0;
	for (auto i = includes.begin(); i != includes.end(); i++) incl |= m_flags[i->c_str()];
	for (auto i = excludes.begin(); i != excludes.end(); i++) excl |= m_flags[i->c_str()];
	if (includes.empty() && excludes.empty()) incl = USHRT_MAX; // include all
	dtQueryFilter filter;
	filter.setIncludeFlags(incl);
	filter.setExcludeFlags(excl);
	for (int i = 0; i < m_areas.defaultCosts.size(); i++) filter.setAreaCost(i, m_areas.defaultCosts[i]);
	return filter;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::GetPoly                                                  |
//  |  Returns a polygon pointer given its detour reference.                LH2'19|
//  +-----------------------------------------------------------------------------+
const dtPoly* NavMeshNavigator::GetPoly(dtPolyRef ref) const
{
	const dtMeshTile* tile; const dtPoly* poly;
	if (!dtStatusFailed(m_navmesh->getTileAndPolyByRef(ref, &tile, &poly)))
		return poly;
	else return 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::CreateNavMeshQuery                                       |
//  |  Creates NavMesQuery from a navmesh and checks for errors.            LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshNavigator::CreateNavMeshQuery()
{
	if (!m_navmesh)
		DETOUR_ERROR(NavMeshStatus::DT | NavMeshStatus::MEM, "NavMeshQuery creation failed: m_navmesh is nullptr\n");
	m_query = dtAllocNavMeshQuery();
	if (!m_query)
		DETOUR_ERROR(NavMeshStatus::DT | NavMeshStatus::MEM, "NavMesh Query could not be allocated\n");
	dtStatus status = m_query->init(m_navmesh, DETOUR_MAX_NAVMESH_NODES);
	if (dtStatusFailed(status))
	{
		dtFreeNavMeshQuery(m_query);
		DETOUR_ERROR(NavMeshStatus::DT | NavMeshStatus::MEM, "Could not init Detour navmesh query\n");
	}
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshNavigator::Load                                                     |
//  |  Loads a navmesh from storage and initializes the query.              LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus NavMeshNavigator::Load(const char* dir, const char* ID)
{
	if (m_navmesh) Clean();
	std::string configfile = std::string(dir) + ID + PF_NAVMESH_CONFIG_FILE_EXTENTION;
	NavMeshConfig config;
	NavMeshStatus status = DeserializeConfigurations(configfile.c_str(), config);
	if (status.Failed()) return status;
	m_flags = config.m_flags;
	m_areas = config.m_areas;
	status = DeserializeNavMesh(dir, ID, m_navmesh);
	if (status.Failed()) return status;
	m_owner = true;
	status = CreateNavMeshQuery();
	return status;
}

} // namespace Lighthouse2

// EOF