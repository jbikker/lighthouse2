/* agent.h - Copyright 2019 Utrecht University
   
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

#include <stdlib.h>
#include <vector>

#include "system.h"				 // float3
#include "physics_placeholder.h" // RigidBody

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  Agent                                                                      |
//  |  Class description of an AI controlled agent.                         LH2'19|
//  +-----------------------------------------------------------------------------+
class Agent
{
public:
	Agent() { m_alive = false; };
	Agent(NavMeshNavigator* navmesh, RigidBody* rb, int maxPathSize)
		: m_rb(rb), m_navmesh(navmesh), m_maxPathCount(maxPathSize)
	{
		m_path.reserve(maxPathSize);
		m_path.resize(maxPathSize); // count is kept using m_pathCount
		m_pathCount = m_targetIdx = 0;
		m_filter = navmesh->GetFilter({}, {});
	};
	Agent(Agent&&) = default;
	Agent& operator=(Agent&&) = default;
	~Agent() {};

	void SetTarget(float3* target) { if (m_pathEndOwner && m_pathEnd) delete m_pathEnd; m_pathEnd = target; m_pathEndOwner = false; }
	void SetTarget(float3 target) { if (m_pathEndOwner && m_pathEnd) *m_pathEnd = target; else m_pathEnd =  new float3(target);  m_pathEndOwner = true; };
	void SetFilter(dtQueryFilter filter) { m_filter = filter; };
	bool UpdateMovement(float deltaTime);
	bool UpdateNavigation(float deltaTime);
	void Clean() { m_pathCount = 0; m_alive = false; };

	void Kill() { m_alive = false; m_rb->Kill(); };
	bool isAlive() const { return m_alive; };

	dtQueryFilter* GetFilter() { return &m_filter; };
	mat4 GetTransform() const { return m_rb->GetTransform(); };
	const float3* GetPos() const { return &m_rb->m_pos; };
	const float3* GetDir() const { return &m_moveDir; };
	float3* GetTarget() const { return m_pathEnd; };
	const std::vector<NavMeshNavigator::PathNode>* GetPath() const { if (m_pathEnd) return &m_path; else return 0; };

	const RigidBody* GetRB() const { return m_rb; };

protected:
	RigidBody* m_rb;
	float3 m_moveDir = { 0.0f, 0.0f, 0.0f };
	float m_nextTarDist = 0.0f; // distance to the next target
	float m_maxLinVel = 6.0f, m_maxLinAcc = 1.0f;
	float m_arrival = 2.5f; // distance at which to slow down
	float m_targetReached = 1.0f; // distance at which to switch to the next target

	NavMeshNavigator* m_navmesh;
	dtQueryFilter m_filter;							// traversal costs and polygon restrictions
	std::vector<NavMeshNavigator::PathNode> m_path; // should NEVER reallocate, invalidates pointers
	float3* m_pathEnd = 0;							// final target, also indicates if it should move at all
	int m_maxPathCount;				// maximum number of path nodes the agent can hold
	int m_pathCount; // number of calculated targets in path array
	int m_targetIdx; // path array index of current target
	bool m_alive = true;			// whether this agent exists
	bool m_pathEndOwner = false;	// whether m_pathEnd is owned by this instance or given
	bool m_onOMC = false;			// whether the Agent is currently traversing an OMC
	bool m_reachable = false; // whether a path to the given end target seems possible at this point

	inline float3 SteeringSeek() const { return m_moveDir * m_maxLinVel; };
	inline float3 SteeringArrival() const { return m_moveDir * m_maxLinVel * (m_nextTarDist / m_arrival); };
	inline float3 SteeringStop() const { return float3{ 0, 0, 0 }; };

private:
	// Move-only (m_rb reference is killed on Kill())
	Agent(const Agent& a);
	Agent& operator=(const Agent& a) = delete;
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshAgents                                                              |
//  |  Class responsible for all agents objects.                            LH2'19|
//  +-----------------------------------------------------------------------------+
class NavMeshAgents
{
public:
	NavMeshAgents(int maxAgents, int maxAgentPathSize, float updateTimeInterval)
		: m_maxAgents(maxAgents), m_maxPathSize(maxAgentPathSize), m_updateTimeInterval(updateTimeInterval)
	{
		m_agents.reserve(maxAgents);
		m_agents.resize(maxAgents);
	};
	~NavMeshAgents() {};

	Agent* AddAgent(NavMeshNavigator* navmesh, RigidBody* rb);
	void RemoveAgent(Agent* agent);
	bool UpdateAgentMovement(float deltaTime);
	bool UpdateAgentBehavior(float deltaTime);
	void Clean();

protected:
	std::vector<Agent> m_agents; // should NEVER reallocate, invalidates pointers
	int m_maxAgents, m_maxPathSize, m_agentCount = 0;
	std::vector<int> m_removedIdx;
	float m_updateTimeInterval, m_timeCounter = 0;

private:
	// not meant to be copied
	NavMeshAgents(const NavMeshAgents&) = delete;
	NavMeshAgents& operator=(const NavMeshAgents&) = delete;
};

} // namespace lighthouse2

// EOF