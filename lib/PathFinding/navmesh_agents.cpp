/* agent.cpp - Copyright 2019 Utrecht University
   
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
#include "navmesh_agents.h"

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  Agent::UpdateMovement                                                      |
//  |  Called after every physics update to update the direction.           LH2'19|
//  +-----------------------------------------------------------------------------+
bool Agent::UpdateMovement(float deltaTime)
{
	if (!m_pathEnd || !m_pathCount)
	{
		// If the path ends, come to a halt
		float3 steering = SteeringStop() - m_rb->m_vel;
		float steerLen = length(steering);
		if (steerLen > m_maxLinAcc) steering *= m_maxLinAcc / steerLen;
		m_rb->AddImpulse(steering);
		return false;
	}
	
	// Check distance to target
	m_moveDir = m_path[m_targetIdx].pos - m_rb->m_pos;
	m_nextTarDist = length(m_moveDir);
	const dtPoly* poly = m_navmesh->GetPoly(m_path[m_targetIdx].poly);

	// When OMC radius reached, teleport to OMC start
	if (!m_onOMC && poly &&
		poly->getType() == DT_POLYTYPE_OFFMESH_CONNECTION &&
		m_nextTarDist <= m_navmesh->GetOMC(m_path[m_targetIdx].poly)->rad)
	{
		m_rb->m_pos = m_path[m_targetIdx].pos;
		m_nextTarDist = 0;
	}

	// When current target reached
	if (m_nextTarDist < m_targetReached)
	{
		// if agent was traversing an OMC, it just finished doing so
		if (m_onOMC) m_onOMC = false;
		// if not, but the reached poly is an OMC, it is now traversing the OMC
		else if (poly &&  poly->getType() == DT_POLYTYPE_OFFMESH_CONNECTION)
			m_onOMC = true;

		if (m_targetIdx < m_pathCount - 1) // if there's another target
		{
			m_targetIdx++; // next path target
			m_moveDir = m_path[m_targetIdx].pos - m_rb->m_pos;
			m_nextTarDist = length(m_moveDir);
		}
		else // if there are no targets left
		{
			for (int i = 0; i < m_pathCount-1; i++) m_path[i] = m_path[m_pathCount-1]; // set all path nodes to target
			if (length(m_path[m_pathCount - 1].pos - *m_pathEnd) < m_targetReached)
			{
				if (m_pathEndOwner) delete m_pathEnd;
				m_pathEnd = 0; // final target reached
			}
		}
	}
	m_moveDir = m_moveDir / m_nextTarDist;

	// Determine the desired velocity
	float3 desiredVelocity;
	if (m_nextTarDist <= m_arrival) desiredVelocity = SteeringArrival();
	else desiredVelocity = SteeringSeek();

	// Determine best feasible impulse
	float3 steering = desiredVelocity - m_rb->m_vel;
	float steerLen = length(steering);
	if (steerLen > m_maxLinAcc) steering *= m_maxLinAcc / steerLen;
	m_rb->AddImpulse(steering);

	return true;
};

//  +-----------------------------------------------------------------------------+
//  |  Agent::UpdateNavigation                                                    |
//  |  Called every AI tick to recalculate the path.                        LH2'19|
//  +-----------------------------------------------------------------------------+
bool Agent::UpdateNavigation(float deltaTime)
{
	if (!m_pathEnd) return false;
	float3 tmpPos = (m_onOMC ? m_path[m_targetIdx].pos : m_rb->m_pos); // can't turn back on an OMC
	if (m_navmesh->FindPathConstSize(tmpPos, *m_pathEnd, m_path.data(), m_pathCount, m_reachable, m_maxPathCount, &m_filter).Success())
		m_targetIdx = 0;
	for (int i = m_pathCount; i < m_maxPathCount; i++) m_path[i] = m_path[m_pathCount-1];
	return true;
}





//  +-----------------------------------------------------------------------------+
//  |  NavMeshAgents::AddAgent                                                    |
//  |  Adds an agent.                                                       LH2'19|
//  +-----------------------------------------------------------------------------+
Agent* NavMeshAgents::AddAgent(NavMeshNavigator* navmesh, RigidBody* rb)
{
	int idx;
	if (m_removedIdx.empty()) // no array holes
	{
		if (m_agentCount < m_maxAgents) idx = m_agentCount++; // capacity at the end
		else
		{   // looking for unknown dead agents
			idx = -1;
			for (int i = 0; i < m_maxAgents; i++) if (!m_agents[i].isAlive())
			{
				idx = i;
				break;
			}
			if (idx == -1) return 0; // none found, capacity full
		}
	}
	else // filling known holes first
	{
		idx = m_removedIdx.back();
		m_removedIdx.pop_back();
	}
	m_agents[idx] = Agent(navmesh, rb, m_maxPathSize);
	return &m_agents[idx];
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshAgents::RemoveAgent                                                 |
//  |  Kills the specified agent.                                           LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshAgents::RemoveAgent(Agent* agent)
{
	agent->Clean();
	int idx = (int)(agent - m_agents.data()) / sizeof(Agent);
	m_removedIdx.push_back(idx);
	agent->Kill();
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshAgents::UpdateAgentMovement                                         |
//  |  Called after every physics tick to add all agent movement impulses.  LH2'19|
//  +-----------------------------------------------------------------------------+
bool NavMeshAgents::UpdateAgentMovement(float deltaTime)
{
	bool changed = false;
	for (std::vector<Agent>::iterator it = m_agents.begin(); it != m_agents.end(); it++)
		if (it->isAlive()) changed |= it->UpdateMovement(deltaTime);
	return changed;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshAgents::UpdateAgentBehavior                                         |
//  |  Called every tick to update all agent plans.                               |
//  |  Only actually updates at the interval given to the constructor.      LH2'19|
//  +-----------------------------------------------------------------------------+
bool NavMeshAgents::UpdateAgentBehavior(float deltaTime)
{
	bool changed = false;
	m_timeCounter += deltaTime;
	if (m_timeCounter < m_updateTimeInterval) return false;
	for (std::vector<Agent>::iterator it = m_agents.begin(); it != m_agents.end(); it++)
		if (it->isAlive()) changed |= it->UpdateNavigation(m_timeCounter);
	m_timeCounter -= m_updateTimeInterval;
	return changed;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshAgents::Clean                                                       |
//  |  Removes all agents.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshAgents::Clean()
{
	for (int i = 0; i < m_agentCount; i++) if (m_agents[i].isAlive()) m_agents[i].Clean();
	m_agentCount = 0;
}

} // namespace lighthouse2

// EOF