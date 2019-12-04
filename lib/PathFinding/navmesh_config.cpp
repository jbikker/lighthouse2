/* navmesh_config.h - Copyright 2019 Utrecht University
   
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

#include "tinyxml2.h" // configuration saving and -loading

#include "system.h" // make_float3
#include "navmesh_common.h"

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig::NavMeshConfig                                               |
//  |  Initializes using default configurations.                            LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshConfig::NavMeshConfig()
{
	m_width = 0;
	m_height = 0;
	m_tileSize = 0;
	m_borderSize = 0;
	m_cs = 1.0f;
	m_ch = 1.0f;
	m_bmin = make_float3(0.0f);
	m_bmax = make_float3(0.0f);

	m_walkableSlopeAngle = 40.0f;
	m_walkableHeight = 10;
	m_walkableClimb = 2;
	m_walkableRadius = 3;

	m_maxEdgeLen = 20;
	m_maxSimplificationError = 2.5f;
	m_minRegionArea = 12;
	m_mergeRegionArea = 25;
	m_maxVertsPerPoly = 6;
	m_detailSampleDist = 10.0f;
	m_detailSampleMaxError = 2.0f;

	m_partitionType = SAMPLE_PARTITION_WATERSHED;
	m_keepInterResults = false;
	m_filterLowHangingObstacles = true;
	m_filterLedgeSpans = true;
	m_filterWalkableLowHeightSpans = true;
	m_id = "default_ID";
	m_printBuildStats = false;
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig::SetAgentInfo                                                |
//  |  Sets all configurations regarding the agent.                         LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshConfig::SetAgentInfo(float angle, int height, int climb, int radius)
{
	m_walkableSlopeAngle = angle;
	m_walkableHeight = height;
	m_walkableClimb = climb;
	m_walkableRadius = radius;
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig::SetPolySettings                                             |
//  |  Sets all configurations regarding the polygon mesh.                  LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshConfig::SetPolySettings(int maxEdgeLen, float maxSimplificationError,
	int minRegionArea, int minMergedRegionArea, int maxVertPerPoly)
{
	m_maxEdgeLen = maxEdgeLen;
	m_maxSimplificationError = maxSimplificationError;
	m_minRegionArea = minRegionArea;
	m_mergeRegionArea = minMergedRegionArea;
	m_maxVertsPerPoly = maxVertPerPoly;
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig::SetDetailPolySettings                                       |
//  |  Sets all configurations regarding the detail polygon mesh.           LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshConfig::SetDetailPolySettings(float sampleDist, float maxSimplificationError)
{
	m_detailSampleDist = sampleDist;
	m_maxSimplificationError = maxSimplificationError;
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig::SetSurfaceFilterSettings                                    |
//  |  Sets what to filter the walkable surfaces for.                       LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshConfig::SetSurfaceFilterSettings(bool lowHangingObstacles,
	bool ledgeSpans, bool WalkableLowHeightSpans)
{
	m_filterLowHangingObstacles = lowHangingObstacles;
	m_filterLedgeSpans = ledgeSpans;
	m_filterWalkableLowHeightSpans = WalkableLowHeightSpans;
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig::ScaleSettings                                               |
//  |  Easily scales all relevant settings when the scene itself is scaled. LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshConfig::ScaleSettings(float scale)
{
	m_cs /= scale; m_ch /= scale;
	m_walkableHeight /= scale; m_walkableClimb /= scale; m_walkableRadius /= scale;
	m_maxEdgeLen /= scale; m_maxSimplificationError /= scale;
	m_minRegionArea /= (scale*scale); m_mergeRegionArea /= (scale*scale);
	m_detailSampleDist /= scale; m_maxSimplificationError /= scale;
};

} // namespace lighthouse2

// EOF