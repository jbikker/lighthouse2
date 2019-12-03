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

#include "navmesh_navigator.h"

namespace lighthouse2 {

#define NAVMESHIO_ERROR(X, ...) return NavMeshError(0, X, "ERROR NavMeshIO: ", __VA_ARGS__)
#define DETOUR_ERROR(X, ...) return NavMeshError(0, X, "ERROR NavMeshNavigator: ", __VA_ARGS__)
#define DETOUR_LOG(...) NavMeshError(0, NavMeshStatus::SUCCESS, "", __VA_ARGS__)

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
		path[0] = PathNode{ start, GetPoly(startRef) };
		path[1] = PathNode{ end, GetPoly(endRef) };
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
		path[i] = PathNode{ straightPath[i], GetPoly(spPolys[i]) };
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
	path[0] = PathNode{ firstPos, GetPoly(startRef) };

	// When start & end are on the same poly
	if (startRef == endRef)
	{
		status = FindClosestPointOnPoly(endRef, end, endPos);
		if (status.Failed()) return status;
		path[1] = PathNode{ endPos, GetPoly(endRef) };
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
		path[i + 1] = PathNode{ iterPos, GetPoly(polyPath[i]) };
	}

	// Finding the closest valid point to the target
	bool pathComplete = (count < maxCount); // means there's room for endPos
	if (pathComplete && (endRef == polyPath[count - 1])) // complete & reachable
	{
		status = FindClosestPointOnPoly(endRef, end, endPos);
		if (status.Failed()) { free(polyPath); return status; }
		path[count + 1] = PathNode{ endPos, GetPoly(endRef) };
		count++;
		reachable = true;
	}
	else if (pathComplete) // path ended, poly unreachable
	{
		status = FindClosestPointOnPoly(polyPath[count - 1], end, endPos);
		if (status.Failed()) { free(polyPath); return status; }
		path[count + 1] = PathNode{ endPos, GetPoly(polyPath[count - 1]) };
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



// Definitions required for NavMesh serialization
static const int NAVMESHSET_MAGIC = 'M' << 24 | 'S' << 16 | 'E' << 8 | 'T'; //'MSET';
static const int NAVMESHSET_VERSION = 1;
struct NavMeshSetHeader { int magic, version, numTiles; dtNavMeshParams params; };
struct NavMeshTileHeader { dtTileRef tileRef; int dataSize; };

//  +-----------------------------------------------------------------------------+
//  |  Serialize                                                                  |
//  |  Writes the navmesh to storage for future use.                        LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus SerializeNavMesh(const char* dir, const char* ID, const dtNavMesh* navMesh)
{

	// Opening navmesh file for writing
	Timer timer;
	std::string filename = std::string(dir) + ID + PF_NAVMESH_FILE_EXTENTION;
	DETOUR_LOG("Saving NavMesh '%s'... ", filename.c_str());
	if (!navMesh) NAVMESHIO_ERROR(NavMeshStatus::INPUT | NavMeshStatus::DT, "Can't serialize '%s', dtNavMesh is nullptr\n", ID);
	FILE* fp;
	fopen_s(&fp, filename.c_str(), "wb");
	if (!fp) NAVMESHIO_ERROR(NavMeshStatus::IO, "Filename '%s' can't be opened\n", filename.c_str());

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

	DETOUR_LOG("%.3fms\n", timer.elapsed());
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  DeserializeNavMesh                                                         |
//  |  Loads a serialized NavMesh from storage and checks for errors.       LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus DeserializeNavMesh(const char* dir, const char* ID, dtNavMesh*& navmesh)
{

	// Opening file
	Timer timer;
	std::string filename = std::string(dir) + ID + PF_NAVMESH_FILE_EXTENTION;
	DETOUR_LOG("Loading NavMesh '%s'... ", filename.c_str());
	if (!FileExists(filename.c_str()))
		NAVMESHIO_ERROR(NavMeshStatus::IO, "NavMesh file '%s' does not exist\n", filename.c_str());
	FILE* fp;
	fopen_s(&fp, filename.c_str(), "rb");
	if (!fp)
		NAVMESHIO_ERROR(NavMeshStatus::IO, "NavMesh file '%s' could not be opened\n", filename.c_str());

	// Reading header
	NavMeshSetHeader header;
	size_t readLen = fread(&header, sizeof(NavMeshSetHeader), 1, fp);
	if (readLen != 1)
	{
		fclose(fp);
		NAVMESHIO_ERROR(NavMeshStatus::IO, "NavMesh file '%s' is corrupted\n", filename.c_str());
	}
	if (header.magic != NAVMESHSET_MAGIC)
	{
		fclose(fp);
		NAVMESHIO_ERROR(NavMeshStatus::IO, "NavMesh file '%s' is corrupted\n", filename.c_str());
	}
	if (header.version != NAVMESHSET_VERSION)
	{
		fclose(fp);
		NAVMESHIO_ERROR(NavMeshStatus::IO, "NavMesh file '%s' has the wrong navmesh set version\n", filename.c_str());
	}

	// Initializing navmesh with header info
	navmesh = dtAllocNavMesh();
	if (!navmesh)
	{
		fclose(fp);
		NAVMESHIO_ERROR(NavMeshStatus::DT | NavMeshStatus::MEM, "NavMesh for '%s' could not be allocated\n", ID);
	}
	dtStatus status = navmesh->init(&header.params);
	if (dtStatusFailed(status))
	{
		fclose(fp);
		dtFreeNavMesh(navmesh);
		NAVMESHIO_ERROR(NavMeshStatus::DT | NavMeshStatus::INIT, "NavMesh for '%s' failed to initialize\n", ID);
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
			NAVMESHIO_ERROR(NavMeshStatus::IO, "NavMesh file '%s' is corrupted\n", filename.c_str());
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
			NAVMESHIO_ERROR(NavMeshStatus::IO, "NavMesh file '%s' is corrupted\n", filename.c_str());
		}
		navmesh->addTile(data, tileHeader.dataSize, DT_TILE_FREE_DATA, tileHeader.tileRef, 0);
	}
	fclose(fp);

	DETOUR_LOG("%.3fms\n", timer.elapsed());
	return NavMeshStatus::SUCCESS;
}

} // namespace Lighthouse2

// EOF