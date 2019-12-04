/* navmesh_common.h - Copyright 2019 Utrecht University

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

#include "system.h" // float3

namespace lighthouse2 {

#define DETOUR_MAX_NAVMESH_NODES 2048
#define DETOUR_MAX_POLYGON_AREA_TYPES 64
#define PF_NAVMESH_FILE_EXTENTION ".navmesh"
#define PF_NAVMESH_CONFIG_FILE_EXTENTION ".navmesh.config"
#define PF_NAVMESH_PMESH_FILE_EXTENTION ".navmesh.pmesh"
#define PF_NAVMESH_DMESH_FILE_EXTENTION ".navmesh.dmesh"
#define PF_NAVMESH_OMC_FILE_EXTENTION ".navmesh.omc"

//  +-----------------------------------------------------------------------------+
//  |  NavMeshErrorCode                                                           |
//  |  Encodes bitwise info about the last error.                                 |
//  |  Error codes can be interpreted by bitwise & operations.              LH2'19|
//  +-----------------------------------------------------------------------------+
struct NavMeshStatus
{
	enum Code
	{
		SUCCESS = 0x0,	// No issues
		RC = 0x1,		// Caused by Recast
		DT = 0x2,		// Caused by Detour
		INPUT = 0x4,	// Incorrect input
		MEM = 0x8,		// Allocation failed, most likely out of memory
		INIT = 0x16,	// A R/D function failed to create a structure
		IO = 0x32		// Issues with I/O
	};
	int code;

	NavMeshStatus() { code = SUCCESS; };
	NavMeshStatus(const int& a) { code = a; };
	NavMeshStatus& operator=(const int& a) { code = a; return *this; };
	operator int() const { return code; };

	bool Success() const { return (code == SUCCESS); };
	bool Failed() const { return (code != SUCCESS); };
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshStatus                                                              |
//  |  Updates the error code of the class, and prints the message to stdout.     |
//  |																			  |
//  |  *internalStatus* is a ptr to the error status of the class, in order       |
//  |  to automatically update the internal status. Use 0 if not needed.          |
//  |  *code* is the NavMeshErrorCode of the error. NMSUCCESS for logging.        |
//  |  *prefix* is an optional string printed before the error message.           |
//  |  *format* and the variadic arguments after that describe the error.         |
//  |																			  |
//  |  The return value is the error code, for chaining purposes.           LH2'19|
//  +-----------------------------------------------------------------------------+
static NavMeshStatus NavMeshError(NavMeshStatus* internalStatus, NavMeshStatus code, const char* prefix, const char* format, ...)
{
	if (internalStatus && code.Failed()) *internalStatus = code;
	printf(prefix);

	va_list ap;
	__crt_va_start(ap, format);
	vprintf(format, ap);
	__crt_va_end(ap);

	return code;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshFlagMapping                                                         |
//  |  Resolves std::string flag labels to short int flags.                 LH2'19|
//  +-----------------------------------------------------------------------------+
struct NavMeshFlagMapping
{
	static const char maxFlags = sizeof(unsigned short) * 8;
	std::vector<std::string> labels;
	void AddFlag(std::string label)
	{
		for (int i = 0; i < labels.size(); i++) if (labels[i] == label)
		{
			NavMeshError(0, NavMeshStatus::INIT, "ERROR NavMeshFlagMapping: ",
				"flag '%s' already exists\n", label.c_str());
			return;
		}
		if ((char)labels.size() < maxFlags) return labels.push_back(label);
		NavMeshError(0, NavMeshStatus::INIT, "ERROR NavMeshFlagMapping: ",
			"has no more capacity for flag '%s'\n", label.c_str());
	};
	void RemoveFlag(std::string label)
	{
		for (auto i = labels.begin(); i != labels.end(); i++) if (*i == label) { labels.erase(i); return; }
		NavMeshError(0, NavMeshStatus::INPUT, "ERROR NavMeshFlagMapping: ",
			"NavMesh flag '%s' does not exist\n", label.c_str());
	}
	unsigned short operator[](const char* label) const
	{
		for (int i = 0; i < maxFlags; i++) if (labels[i] == label) return 0x1 << i;
		NavMeshError(0, NavMeshStatus::INPUT, "ERROR NavMeshFlagMapping: ",
			"NavMesh flag '%s' does not exist\n", label);
		return 0;
	};
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshAreaMapping                                                         |
//  |  Resolves std::string area labels to char values.                     LH2'19|
//  +-----------------------------------------------------------------------------+
struct NavMeshAreaMapping
{
	static const int maxAreas = DETOUR_MAX_POLYGON_AREA_TYPES;
	std::vector<std::string> labels = { "DEFAULT_AREA_TYPE" };
	std::vector<float> defaultCosts = { 1.0f };
	void AddArea(std::string label, float defaultCost = 1.0f)
	{
		for (int i = 0; i < labels.size(); i++) if (labels[i] == label)
		{
			NavMeshError(0, NavMeshStatus::INIT, "ERROR NavMeshAreaMapping: ",
				"area type '%s' already exists\n", label.c_str());
			return;
		}
		if ((int)labels.size() < maxAreas)
		{
			labels.push_back(label);
			defaultCosts.push_back(defaultCost);
			return;
		}
		NavMeshError(0, NavMeshStatus::INIT, "ERROR NavMeshAreaMapping: ",
			"has no more capacity for area type '%s'\n", label.c_str());
	}
	void RemoveArea(std::string label)
	{
		for (int i = 0; i < labels.size(); i++) if (labels[i] == label)
		{
			labels.erase(labels.begin() + i);
			defaultCosts.erase(defaultCosts.begin() + i);
			return;
		}
		NavMeshError(0, NavMeshStatus::INPUT, "ERROR NavMeshAreaMapping: ",
			"NavMesh area type '%s' does not exist\n", label.c_str());
	}
	void SetDefaultCost(std::string label, float defaultCost)
	{
		for (int i = 0; i < labels.size(); i++) if (labels[i] == label)
		{
			defaultCosts[i] = defaultCost;
			return;
		}
		NavMeshError(0, NavMeshStatus::INPUT, "ERROR NavMeshAreaMapping: ",
			"NavMesh area type '%s' does not exist\n", label.c_str());
	}
	unsigned char operator[](const char* label) const
	{
		for (int i = 0; i < maxAreas; i++) if (labels[i] == label) return i;
		NavMeshError(0, NavMeshStatus::INPUT, "ERROR NavMeshAreaMapping: ",
			"NavMesh area type '%s' does not exist\n", label);
		return 0;
	};
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig                                                              |
//  |  Contains all settings regarding the navmesh generation.              LH2'19|
//  +-----------------------------------------------------------------------------+
struct NavMeshConfig
{

	int m_width, m_height, m_tileSize, m_borderSize;		 // Automatically computed
	float m_cs, m_ch;										 // Voxel cell size and -height
	float3 m_bmin, m_bmax;									 // AABB navmesh restraints
	float m_walkableSlopeAngle;								 // In degrees
	int m_walkableHeight, m_walkableClimb, m_walkableRadius; // In voxels
	int m_maxEdgeLen;
	float m_maxSimplificationError;
	int m_minRegionArea, m_mergeRegionArea, m_maxVertsPerPoly; // maxVertsPerPoly should not exceed 6
	float m_detailSampleDist, m_detailSampleMaxError;

	//  +-------------------------------------------------------------------------------------------------------+
	//  |  SamplePartitionType                                                                                  |
	//  |  The heightfield is partitioned so that a simple algorithm can triangulate the walkable areas.	    |
	//  |  There are 3 martitioning methods, each with some pros and cons:										|
	//  |  1) Watershed partitioning																			|
	//  |    - the classic Recast partitioning																	|
	//  |    - creates the nicest tessellation																	|
	//  |    - usually slowest																					|
	//  |    - partitions the heightfield into nice regions without holes or overlaps							|
	//  |    - the are some corner cases where this method creates produces holes and overlaps					|
	//  |       - holes may appear when a small obstacle is close to large open area (triangulation won't fail) |
	//  |       - overlaps may occur on narrow spiral corridors (i.e stairs) and triangulation may fail         |
	//  |    * generally the best choice if you precompute the nacmesh, use this if you have large open areas	|
	//  |  2) Monotone partioning																				|
	//  |    - fastest																							|
	//  |    - partitions the heightfield into regions without holes and overlaps (guaranteed)					|
	//  |    - creates long thin polygons, which sometimes causes paths with detours							|
	//  |    * use this if you want fast navmesh generation														|
	//  |  3) Layer partitoining																				|
	//  |    - quite fast																						|
	//  |    - partitions the heighfield into non-overlapping regions											|
	//  |    - relies on the triangulation code to cope with holes (thus slower than monotone partitioning)		|
	//  |    - produces better triangles than monotone partitioning												|
	//  |    - does not have the corner cases of watershed partitioning											|
	//  |    - can be slow and create a bit ugly tessellation (still better than monotone)						|
	//  |      if you have large open areas with small obstacles (not a problem if you use tiles)				|
	//  |    * good choice to use for tiled navmesh with medium and small sized tiles					  LH2'19|
	//  +-------------------------------------------------------------------------------------------------------+
	enum SamplePartitionType
	{
		SAMPLE_PARTITION_WATERSHED,
		SAMPLE_PARTITION_MONOTONE,
		SAMPLE_PARTITION_LAYERS,
	};

	SamplePartitionType m_partitionType; // The partitioning method
	bool m_keepInterResults;			 // Holding on to the intermediate data structures
	bool m_filterLowHangingObstacles;	 // Filtering for low obstacles
	bool m_filterLedgeSpans;			 // Filtering for ledges
	bool m_filterWalkableLowHeightSpans; // Filtering for low ceilings
	bool m_printBuildStats;				 // Printing detailed build time statistics
	std::string m_id;					 // Name of the navmesh instance
	NavMeshFlagMapping m_flags;			 // The named polygon flags
	NavMeshAreaMapping m_areas;			 // The named polygon area types

	NavMeshConfig();

	void SetCellSize(float width, float height) { m_cs = width; m_ch = height; };
	void SetAABB(float3 min, float3 max) { m_bmin = min; m_bmax = max; }; // if AABB is not 3D, input mesh is used
	void SetAgentInfo(float maxWalkableAngle, int minWalkableHeight,
		int maxClimbableHeight, int minWalkableRadius);
	void SetPolySettings(int maxEdgeLen, float maxSimplificationError,
		int minRegionArea, int minMergedRegionArea, int maxVertPerPoly);
	void SetDetailPolySettings(float sampleDist, float maxSimplificationError);
	void SetPartitionType(SamplePartitionType type) { m_partitionType = type; };
	void SetKeepInterResults(bool keep) { m_keepInterResults = keep; };
	void SetSurfaceFilterSettings(bool lowHangingObstacles,
		bool ledgeSpans, bool WalkableLowHeightSpans);
	void SetPrintBuildStats(bool print) { m_printBuildStats = print; };
	void SetID(const char* ID) { m_id = ID; };
	void AddFlag(const char* label) { m_flags.AddFlag(label); };
	void AddAreaType(const char* label, float defaultTraversalCost = 0.0f) { m_areas.AddArea(label, defaultTraversalCost); };
	void ScaleSettings(float scale);
};

} // namespace Lighthouse2

// EOF