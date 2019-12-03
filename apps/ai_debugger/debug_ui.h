/* debug_ui.h - Copyright 2019 Utrecht University
   
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

#include "navmesh_navigator.h"
#include "navmesh_shader.h"
#include "navmesh_agents.h"

//  +-----------------------------------------------------------------------------+
//  |  PathDrawingTool                                                            |
//  |  Handles manual path drawing.                                         LH2'19|
//  +-----------------------------------------------------------------------------+
class PathDrawingTool
{
public:
	PathDrawingTool(NavMeshShader* shader, NavMeshNavigator*& navmesh)
		: m_shader(shader), m_navmesh(navmesh) {};
	~PathDrawingTool() {};

	//  +-----------------------------------------------------------------------------+
	//  |  PathDrawingTool::SetStart                                                  |
	//  |  Sets the start of the path.                                          LH2'19|
	//  +-----------------------------------------------------------------------------+
	void SetStart(float3 pos)
	{
		m_v0 = pos;
		m_vertSet |= V0SET;
		m_shader->SetPathStart(&m_v0);
		if (m_vertSet == BOTHSET && m_navmesh)
			if (m_navmesh->FindPath(m_v0, m_v1, m_path, m_reachable).Success())
				m_shader->SetPath(&m_path);
	}

	//  +-----------------------------------------------------------------------------+
	//  |  PathDrawingTool::SetEnd                                                    |
	//  |  Sets the end of the path.                                            LH2'19|
	//  +-----------------------------------------------------------------------------+
	void SetEnd(float3 pos)
	{
		m_v1 = pos;
		m_vertSet |= V1SET;
		m_shader->SetPathEnd(&m_v1);
		if (m_vertSet == BOTHSET && m_navmesh)
			if (m_navmesh->FindPath(m_v0, m_v1, m_path, m_reachable).Success())
				m_shader->SetPath(&m_path);
	}

	//  +-----------------------------------------------------------------------------+
	//  |  PathDrawingTool::Clear                                                     |
	//  |  Resets the internal state and removes the path from the shader.      LH2'19|
	//  +-----------------------------------------------------------------------------+
	void Clear()
	{
		if (m_vertSet != NONESET)
		{
			m_vertSet = NONESET;
			m_shader->SetPath(0);
			m_shader->SetPathStart(0);
			m_shader->SetPathEnd(0);
			m_path.clear();
			m_v0 = m_v1 = float3{ 0, 0, 0 };
		}
	}

	const float3* GetStart() const { return &m_v0; };
	const float3* GetEnd() const { return &m_v1; };
	const bool* GetReachable() const { return &m_reachable; };

protected:
	NavMeshShader* m_shader;
	NavMeshNavigator*& m_navmesh; // navMeshNavigator in main_ui can reallocate

	float3 m_v0 = float3{ 0, 0, 0 }, m_v1 = float3{ 0, 0, 0 };
	std::vector<NavMeshNavigator::PathNode> m_path;
	bool m_reachable = false;
	enum { V0SET = 0x1, V1SET = 0x2, BOTHSET = 0x3, NONESET = 0x0 };
	unsigned char m_vertSet = NONESET;
};

//  +-----------------------------------------------------------------------------+
//  |  AgentNavigationTool                                                        |
//  |  Handles selecting- and editing agents.                               LH2'19|
//  +-----------------------------------------------------------------------------+
class AgentNavigationTool
{
public:
	AgentNavigationTool(NavMeshShader* shader, NavMeshNavigator*& navmesh, PathDrawingTool* pathTool, SELECTIONTYPE* selectionType)
		: m_shader(shader), m_navmesh(navmesh), m_pathTool(pathTool), m_selectionType(selectionType) {};
	~AgentNavigationTool() {};

	//  +-----------------------------------------------------------------------------+
	//  |  AgentNavigationTool::SelectAgent                                           |
	//  |  Selects an agent, highlights the instance, and plots its path.       LH2'19|
	//  +-----------------------------------------------------------------------------+
	void SelectAgent(int InstID)
	{
		Clear();
		m_agent = m_shader->SelectAgent(InstID);
		if (!m_agent) return;
		*m_selectionType = SELECTION_AGENT;

		// set polygon filter
		TwDefine(" Debugging/flags visible=true ");
		for (int i = 0; i < NavMeshFlagMapping::maxFlags; i++)
		{
			m_filterFlags[i] = (m_agent->GetFilter()->getIncludeFlags() & (0x1 << i));
			if (m_agent->GetFilter()->getExcludeFlags() & (0x1 << i)) m_filterFlags[i] = false;
		}
		TwDefine(" Debugging/'area cost' visible=true ");
		for (int i = 0; i < NavMeshAreaMapping::maxAreas; i++)
			m_areaCosts[i] = m_agent->GetFilter()->getAreaCost(i);

		// shade excluded polygons
		unsigned short includes = 0;
		for (int i = 0; i < NavMeshFlagMapping::maxFlags; i++)
			if (m_filterFlags[i]) includes |= (0x1 << i);
		m_shader->SetExcludedPolygons(m_navmesh, ~includes);

		if (m_agent->GetTarget()) // if it's already moving
		{
			m_pathSet = true;
			m_pathTool->SetStart(*m_agent->GetPos());
			m_pathTool->SetEnd(*m_agent->GetTarget());
			m_shader->SetPath(m_agent->GetPath());
		}
		else // if it's standing still
		{
			m_pathSet = false;
			m_pathTool->Clear();
		}
	}

	//  +-----------------------------------------------------------------------------+
	//  |  AgentNavigationTool::SetTarget                                             |
	//  |  Assigns a target the the selected agent and updates its navigation.  LH2'19|
	//  +-----------------------------------------------------------------------------+
	void SetTarget(float3 pos)
	{
		if (!m_agent) return;
		m_pathSet = true;
		m_agent->SetTarget(pos);
		m_agent->UpdateNavigation(0);
		m_pathTool->SetStart(*m_agent->GetPos());
		m_pathTool->SetEnd(pos);
		m_shader->SetPath(m_agent->GetPath());
	}

	//  +-----------------------------------------------------------------------------+
	//  |  AgentNavigationTool::RemoveSelectedAgent                                   |
	//  |  Removes the selected agent.                                          LH2'19|
	//  +-----------------------------------------------------------------------------+
	void RemoveSelectedAgent()
	{
		if (!m_agent) return;
		m_shader->RemoveAgentFromScene(m_agent);
		m_agent->Kill();
		Clear();
	}

	//  +-----------------------------------------------------------------------------+
	//  |  AgentNavigationTool::Clear                                                 |
	//  |  Resets the internal state, stops highlighting, and removes path.     LH2'19|
	//  +-----------------------------------------------------------------------------+
	void Clear()
	{
		if (m_agent) // if there was an agent selected at all
		{
			m_shader->Deselect();
			m_shader->SetExcludedPolygons(0, 0);
			m_pathTool->Clear();
			*m_selectionType = SELECTION_NONE;
		}
		m_agent = 0;
		m_pathSet = false;
		TwDefine(" Debugging/flags visible=false ");
		TwDefine(" Debugging/'area cost' visible=false ");
	}

	void ApplyChanges()
	{
		if (m_agent)
		{
			// Applying filter changes
			dtQueryFilter* filter = m_agent->GetFilter();
			unsigned short includes = 0;
			for (int i = 0; i < NavMeshFlagMapping::maxFlags; i++) 
				if (m_filterFlags[i]) includes |= (0x1 << i);
			filter->setIncludeFlags(includes);
			filter->setExcludeFlags(~includes);
			m_shader->SetExcludedPolygons(m_navmesh, ~includes);
			for (int i = 0; i < NavMeshAreaMapping::maxAreas; i++)
				filter->setAreaCost(i, m_areaCosts[i]);
		}
	}

	bool* GetFilterFlag(int idx) { return &m_filterFlags[idx]; };
	float* GetAreaCost(int idx) { return &m_areaCosts[idx]; };

protected:
	NavMeshShader* m_shader;
	NavMeshNavigator*& m_navmesh;
	PathDrawingTool* m_pathTool;
	SELECTIONTYPE* m_selectionType;

	Agent* m_agent = 0;
	bool m_pathSet = false;
	bool m_filterFlags[NavMeshFlagMapping::maxFlags];
	float m_areaCosts[NavMeshAreaMapping::maxAreas];
};

// EOF