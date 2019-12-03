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

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig::Save                                                        |
//  |  Writes the configurations to storage as an XML.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshConfig::Save(const char* filename) const
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLNode* root = doc.NewElement("configurations");
	doc.InsertFirstChild(root);

	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("width")))->SetText(m_width);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("height")))->SetText(m_height);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("tileSize")))->SetText(m_tileSize);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("borderSize")))->SetText(m_borderSize);

	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("cs")))->SetText(m_cs);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("ch")))->SetText(m_ch);
	tinyxml2::XMLNode* bmin = doc.NewElement("bmin");
	root->InsertEndChild(bmin);
		((tinyxml2::XMLElement*)bmin->InsertEndChild(doc.NewElement("x")))->SetText(m_bmin.x);
		((tinyxml2::XMLElement*)bmin->InsertEndChild(doc.NewElement("y")))->SetText(m_bmin.y);
		((tinyxml2::XMLElement*)bmin->InsertEndChild(doc.NewElement("z")))->SetText(m_bmin.z);
	tinyxml2::XMLNode* bmax = doc.NewElement("bmax");
	root->InsertEndChild(bmax);
		((tinyxml2::XMLElement*)bmax->InsertEndChild(doc.NewElement("x")))->SetText(m_bmax.x);
		((tinyxml2::XMLElement*)bmax->InsertEndChild(doc.NewElement("y")))->SetText(m_bmax.y);
		((tinyxml2::XMLElement*)bmax->InsertEndChild(doc.NewElement("z")))->SetText(m_bmax.z);

	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("walkableSlopeAngle")))->SetText(m_walkableSlopeAngle);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("walkableClimb")))->SetText(m_walkableClimb);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("walkableHeight")))->SetText(m_walkableHeight);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("walkableRadius")))->SetText(m_walkableRadius);

	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("maxEdgeLen")))->SetText(m_maxEdgeLen);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("maxSimplificationError")))->SetText(m_maxSimplificationError);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("minRegionArea")))->SetText(m_minRegionArea);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("mergeRegionArea")))->SetText(m_mergeRegionArea);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("maxVertsPerPoly")))->SetText(m_maxVertsPerPoly);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("detailSampleDist")))->SetText(m_detailSampleDist);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("detailSampleMaxError")))->SetText(m_detailSampleMaxError);

	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("partitionType")))->SetText(m_partitionType);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("keepInterResults")))->SetText(m_keepInterResults);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("filterLowHangingObstacles")))->SetText(m_filterLowHangingObstacles);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("filterLedgeSpans")))->SetText(m_filterLedgeSpans);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("filterWalkableLowHeightSpans")))->SetText(m_filterWalkableLowHeightSpans);

	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("printBuildStats")))->SetText(m_printBuildStats);
	((tinyxml2::XMLElement*)root->InsertEndChild(doc.NewElement("ID")))->SetText(m_id.c_str());

	tinyxml2::XMLNode* areas = doc.NewElement("areas");
	root->InsertEndChild(areas);
		((tinyxml2::XMLElement*)areas->InsertEndChild(doc.NewElement("count")))->SetText(m_areas.labels.size());
		tinyxml2::XMLNode* arealabels = doc.NewElement("labels");
		areas->InsertEndChild(arealabels);
			for (auto l : m_areas.labels)
				((tinyxml2::XMLElement*)arealabels->InsertEndChild(doc.NewElement("item")))->SetText(l.c_str());
		tinyxml2::XMLNode* defaultCosts = doc.NewElement("defaultCosts");
		areas->InsertEndChild(defaultCosts);
			for (auto c : m_areas.defaultCosts)
				((tinyxml2::XMLElement*)defaultCosts->InsertEndChild(doc.NewElement("item")))->SetText(c);
	tinyxml2::XMLNode* flags = doc.NewElement("flags");
	root->InsertEndChild(flags);
		((tinyxml2::XMLElement*)flags->InsertEndChild(doc.NewElement("count")))->SetText(m_flags.labels.size());
		tinyxml2::XMLNode* flaglabels = doc.NewElement("labels");
		flags->InsertEndChild(flaglabels);
		for (auto l : m_flags.labels)
			((tinyxml2::XMLElement*)flaglabels->InsertEndChild(doc.NewElement("item")))->SetText(l.c_str());

	doc.SaveFile(filename);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshConfig::Load                                                        |
//  |  Loads an XML configurations file from storage.                       LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus NavMeshConfig::Load(const char* filename)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLError result = doc.LoadFile(filename);
	if (result != tinyxml2::XML_SUCCESS)
		return NavMeshError(0, NavMeshStatus::IO, "", "Config file '%s' could not be opened\n", filename);
	tinyxml2::XMLNode* root = doc.FirstChildElement("configurations");
	if (root == nullptr)
		return NavMeshError(0, NavMeshStatus::INIT, "", "tinyXML2 errored while loading config file '%s'\n", filename);

	if (root->FirstChildElement("width")) root->FirstChildElement("width")->QueryIntText(&m_width);
	if (root->FirstChildElement("height")) root->FirstChildElement("height")->QueryIntText(&m_height);
	if (root->FirstChildElement("tileSize")) root->FirstChildElement("tileSize")->QueryIntText(&m_tileSize);
	if (root->FirstChildElement("borderSize")) root->FirstChildElement("borderSize")->QueryIntText(&m_borderSize);

	if (root->FirstChildElement("cs")) root->FirstChildElement("cs")->QueryFloatText(&m_cs);
	if (root->FirstChildElement("ch")) root->FirstChildElement("ch")->QueryFloatText(&m_ch);
	if (root->FirstChildElement("bmin")) root->FirstChildElement("bmin")->FirstChildElement("x")->QueryFloatText(&m_bmin.x);
	if (root->FirstChildElement("bmin")) root->FirstChildElement("bmin")->FirstChildElement("y")->QueryFloatText(&m_bmin.y);
	if (root->FirstChildElement("bmin")) root->FirstChildElement("bmin")->FirstChildElement("z")->QueryFloatText(&m_bmin.z);
	if (root->FirstChildElement("bmax")) root->FirstChildElement("bmax")->FirstChildElement("x")->QueryFloatText(&m_bmax.x);
	if (root->FirstChildElement("bmax")) root->FirstChildElement("bmax")->FirstChildElement("y")->QueryFloatText(&m_bmax.y);
	if (root->FirstChildElement("bmax")) root->FirstChildElement("bmax")->FirstChildElement("z")->QueryFloatText(&m_bmax.z);

	if (root->FirstChildElement("walkableSlopeAngle")) root->FirstChildElement("walkableSlopeAngle")->QueryFloatText(&m_walkableSlopeAngle);
	if (root->FirstChildElement("walkableHeight")) root->FirstChildElement("walkableHeight")->QueryIntText(&m_walkableHeight);
	if (root->FirstChildElement("walkableClimb")) root->FirstChildElement("walkableClimb")->QueryIntText(&m_walkableClimb);
	if (root->FirstChildElement("walkableRadius")) root->FirstChildElement("walkableRadius")->QueryIntText(&m_walkableRadius);

	if (root->FirstChildElement("maxEdgeLen")) root->FirstChildElement("maxEdgeLen")->QueryIntText(&m_maxEdgeLen);
	if (root->FirstChildElement("maxSimplificationError")) root->FirstChildElement("maxSimplificationError")->QueryFloatText(&m_maxSimplificationError);
	if (root->FirstChildElement("minRegionArea")) root->FirstChildElement("minRegionArea")->QueryIntText(&m_minRegionArea);
	if (root->FirstChildElement("mergeRegionArea")) root->FirstChildElement("mergeRegionArea")->QueryIntText(&m_mergeRegionArea);
	if (root->FirstChildElement("maxVertsPerPoly")) root->FirstChildElement("maxVertsPerPoly")->QueryIntText(&m_maxVertsPerPoly);
	if (root->FirstChildElement("detailSampleDist")) root->FirstChildElement("detailSampleDist")->QueryFloatText(&m_detailSampleDist);
	if (root->FirstChildElement("detailSampleMaxError")) root->FirstChildElement("detailSampleMaxError")->QueryFloatText(&m_detailSampleMaxError);

	if (root->FirstChildElement("partitionType")) root->FirstChildElement("partitionType")->QueryIntText((int*)&m_partitionType);
	if (root->FirstChildElement("keepInterResults")) root->FirstChildElement("keepInterResults")->QueryBoolText(&m_keepInterResults);
	if (root->FirstChildElement("filterLowHangingObstacles")) root->FirstChildElement("filterLowHangingObstacles")->QueryBoolText(&m_filterLowHangingObstacles);
	if (root->FirstChildElement("filterLedgeSpans")) root->FirstChildElement("filterLedgeSpans")->QueryBoolText(&m_filterLedgeSpans);
	if (root->FirstChildElement("filterWalkableLowHeightSpans")) root->FirstChildElement("filterWalkableLowHeightSpans")->QueryBoolText(&m_filterWalkableLowHeightSpans);

	if (root->FirstChildElement("printBuildStats")) root->FirstChildElement("printBuildStats")->QueryBoolText(&m_printBuildStats);
	if (root->FirstChildElement("ID")) m_id = root->FirstChildElement("ID")->FirstChild()->Value();

	// Loading m_area
	tinyxml2::XMLElement* item;
	int areaCount = 0, flagCount = 0;
	if (root->FirstChildElement("areas") && root->FirstChildElement("areas")->FirstChildElement("count"))
		root->FirstChildElement("areas")->FirstChildElement("count")->QueryIntText(&areaCount);
	if (areaCount > 0)
	{
		m_areas.labels.resize(areaCount);
		if (root->FirstChildElement("areas")->FirstChildElement("labels"))
		{
			item = root->FirstChildElement("areas")->FirstChildElement("labels")->FirstChildElement("item");
			for (int i = 0; i < areaCount; i++)
			{
				if (item) m_areas.labels[i] = item->FirstChild()->Value(); else break;
				item = item->NextSiblingElement("item");
			}
		}
		m_areas.defaultCosts.resize(areaCount);
		if (root->FirstChildElement("areas")->FirstChildElement("defaultCosts"))
		{
			item = root->FirstChildElement("areas")->FirstChildElement("defaultCosts")->FirstChildElement("item");
			for (int i = 0; i < areaCount; i++)
			{
				if (item) item->QueryFloatText(&m_areas.defaultCosts[i]); else break;
				item = item->NextSiblingElement("item");
			}
		}
	}

	// Loading m_flags
	if (root->FirstChildElement("flags") && root->FirstChildElement("flags")->FirstChildElement("count"))
		root->FirstChildElement("flags")->FirstChildElement("count")->QueryIntText(&flagCount);
	if (flagCount > 0)
	{
		m_flags.labels.resize(flagCount);
		if (root->FirstChildElement("flags")->FirstChildElement("labels"))
		{
			item = root->FirstChildElement("flags")->FirstChildElement("labels")->FirstChildElement("item");
			for (int i = 0; i < flagCount; i++)
			{
				if (item) m_flags.labels[i] = item->FirstChild()->Value(); else break;
				item = item->NextSiblingElement("item");
			}
		}
	}

	return NavMeshStatus::SUCCESS;
}

} // namespace lighthouse2

// EOF