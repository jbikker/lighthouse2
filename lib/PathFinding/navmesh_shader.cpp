/* navmesh_shader.cpp - Copyright 2019 Utrecht University
   
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

#include "DetourNode.h"

#include "platform.h"       // DrawLineStrip, DrawLines
#include "navmesh_shader.h"

namespace lighthouse2 {

//  +--------------------------------------------------------------------------------------------+
//  |																							 |
//  |			GL Drawing			     														 |
//  |																							 |
//  +--------------------------------------------------------------------------------------------+

//  +-----------------------------------------------------------------------------+
//  |  DrawPolys                                                                  |
//  |  Draws a list of polygons to the screen.                              LH2'19|
//  +-----------------------------------------------------------------------------+
void DrawPolys(const std::vector<const dtPoly*> polys, const std::vector<NavMeshShader::Vert> verts, float4 color, const Camera* camera)
{
	if (!camera) return;
	for (auto p : polys)
	{
		std::vector<float3> world;
		std::vector<float2> screen(p->vertCount);
		std::vector<float4> colors(p->vertCount, color);
		for (int i = 0; i < p->vertCount; i++)
			world.push_back(*verts[p->verts[i]].pos);
		camera->WorldToScreenPos(world.data(), screen.data(), (int)world.size());

		DrawShapeOnScreen(screen, colors, GL_TRIANGLE_FAN);
	}
}

//  +-----------------------------------------------------------------------------+
//  |  DrawVerts                                                                  |
//  |  Draws a list of vertices to the screen.                              LH2'19|
//  +-----------------------------------------------------------------------------+
void DrawVerts(const std::vector<NavMeshShader::Vert> verts, float4 color, float width, const Camera* camera)
{
	int count = (int)verts.size();
	std::vector<float3> world(count);
	std::vector<float2> screen(count);
	std::vector<float4> colors(count, color);
	for (int i = 0; i < count; i++) world[i] = *verts[i].pos;
	camera->WorldToScreenPos(world.data(), screen.data(), count);
	DrawShapeOnScreen(screen, colors, GL_POINTS, width);
}

//  +-----------------------------------------------------------------------------+
//  |  DrawEdges                                                                  |
//  |  Draws a list of edges to the screen.                                 LH2'19|
//  +-----------------------------------------------------------------------------+
void DrawEdges(const std::vector<NavMeshShader::Edge> edges, const std::vector<NavMeshShader::Vert> verts, float4 color, float width, const Camera* camera)
{
	if (!camera) return;
	int count = (int)edges.size();
	std::vector<float3> world(count * 2);
	std::vector<float2> screen(count * 2);
	std::vector<float4> colors(count * 2, color);
	for (int i = 0; i < count; i++)
	{
		world[i * 2] = *verts[edges[i].v1].pos;
		world[i * 2 + 1] = *verts[edges[i].v2].pos;
	}
	camera->WorldToScreenPos(world.data(), screen.data(), count * 2);
	DrawShapeOnScreen(screen, colors, GL_LINES, width);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::DrawGL                                                      |
//  |  Draws all applicable GL shapes on the window.                              |
//  |  Navmesh shading using GL, selection highlighting, agent impulses,          |
//  |  paths, path beacons, and highlighting inecessible polygons.          LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::DrawGL() const
{
	if (m_shadeVerts) DrawVerts(m_verts, m_vertColor, m_vertWidthGL, m_renderer->GetCamera());
	if (m_shadeEdges) DrawEdges(m_edges, m_verts, m_edgeColor, m_edgeWidthGL, m_renderer->GetCamera());
	if (m_shadePolys)
	{
		std::vector<const dtPoly*> polys(m_polys.size());
		for (int i = 0; i < (int)m_polys.size(); i++) if (!m_polys[i].triIDs.empty()) // exclude OMCs
			polys[i] = m_polys[i].poly;
		DrawPolys(polys, m_verts, m_polyColor, m_renderer->GetCamera());
	}

	if (m_polySelect) DrawPolys({ m_polySelect->poly }, m_verts, m_highLightColor, m_renderer->GetCamera());
	if (m_vertSelect) DrawVerts({ *m_vertSelect }, m_highLightColor, m_vertWidthGL, m_renderer->GetCamera());
	if (m_edgeSelect) DrawEdges({ *m_edgeSelect }, m_verts, m_highLightColor, m_edgeWidthGL, m_renderer->GetCamera());
	if (m_agentSelect) DrawAgentHighlightGL();
	if (m_agentSelect) DrawAgentImpulse();

	DrawTmpOMCs();

	if (!m_excludedPolygons.empty()) DrawPolys(m_excludedPolygons, m_verts, m_excludedColor, m_renderer->GetCamera());
	if (m_path && !m_path->empty()) PlotPath();
	if (m_pathStart || m_pathEnd) DrawPathMarkers();
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::DrawAgentHighlightGL                                        |
//  |  Draws a highlighted agent on the screen.                             LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::DrawAgentHighlightGL() const
{
	mat4 transform = m_agentSelect->agent->GetTransform();
	std::vector<float3> world(11);
	std::vector<float4> colors(11, m_highLightColor);
	std::vector<float2> screen(11);
	world[0] = make_float3(transform * float4{ 0, 0, 0, 1.0f });
	world[1] = make_float3(transform * float4{ .5f,  .5f, 0, 1.0f });
	world[2] = make_float3(transform * float4{ -.5f,  .5f, 0, 1.0f });
	world[3] = make_float3(transform * float4{ .5f, -.5f, 0, 1.0f });
	world[4] = make_float3(transform * float4{ -.5f, -.5f, 0, 1.0f });
	world[5] = make_float3(transform * float4{ 0, 0, 0, 1.0f });
	world[6] = make_float3(transform * float4{ 0,  .5f,  .5f, 1.0f });
	world[7] = make_float3(transform * float4{ 0,  .5f, -.5f, 1.0f });
	world[8] = make_float3(transform * float4{ 0, -.5f,  .5f, 1.0f });
	world[9] = make_float3(transform * float4{ 0, -.5f, -.5f, 1.0f });
	world[10] = make_float3(transform * float4{ 0, 0, 0, 1.0f });

	m_renderer->GetCamera()->WorldToScreenPos(world.data(), screen.data(), 11);
	DrawShapeOnScreen(screen, colors, GL_LINE_LOOP, 5.0f);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::DrawAgentImpulse                                            |
//  |  Draws a line indicating the agent's movement.                        LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::DrawAgentImpulse() const
{
	if (!m_agentSelect) return;
	const float4 col = { .1f, .9f, .1f, 1.0f };
	const RigidBody* rb = m_agentSelect->agent->GetRB();
	float3 world[2] = { rb->m_pos, rb->m_pos + rb->m_impulse * m_agentImpulseScale };
	std::vector<float4> colors{ col, col };
	std::vector<float2> screen(2);
	m_renderer->GetCamera()->WorldToScreenPos(world, screen.data(), 2);
	DrawShapeOnScreen(screen, colors, GL_LINES, 5.0f);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::PlotPath                                                    |
//  |  Plots the path calculated before as a series of lines                LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::PlotPath() const
{
	std::vector<float3> world(m_path->size());
	std::vector<float2> screen(m_path->size());
	std::vector<float4> colors(m_path->size(), m_pathColor);
	for (size_t i = 0; i < m_path->size(); i++)
		world[i] = m_path->at(i).pos;

	m_renderer->GetCamera()->WorldToScreenPos(world.data(), screen.data(), (int)m_path->size());
	DrawShapeOnScreen(screen, colors, GL_LINE_STRIP, m_pathWidth);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::DrawPathMarkers                                             |
//  |  Draws the start and end of the path as beacons.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::DrawPathMarkers() const
{
	if (!m_pathStart && !m_pathEnd) return;
	std::vector<float3> world;
	std::vector<float4> color;
	if (m_pathStart)
	{
		world.push_back(*m_pathStart);
		world.push_back(*m_pathStart + m_beaconLen);
		color.push_back(m_beaconStColor);
		color.push_back(float4());
	}
	if (m_pathEnd)
	{
		world.push_back(*m_pathEnd);
		world.push_back(*m_pathEnd + m_beaconLen);
		color.push_back(m_beaconEnColor);
		color.push_back(float4());
	}
	std::vector<float2> screen(world.size());
	m_renderer->GetCamera()->WorldToScreenPos(world.data(), screen.data(), (int)world.size());
	DrawShapeOnScreen(screen, color, GL_LINES, m_beaconWidth);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::DrawTmpOMCs                                                 |
//  |  Draws the OMCs added during runtime before changes are applied.      LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::DrawTmpOMCs() const
{
	if (m_tmpVert.pos) DrawVerts({ m_tmpVert }, m_highLightColor, m_vertWidthGL, m_renderer->GetCamera());
	if (!m_tmpVerts.empty()) DrawVerts(m_tmpVerts, m_highLightColor, m_vertWidthGL, m_renderer->GetCamera());
	if (!m_tmpEdges.empty()) DrawEdges(m_tmpEdges, m_tmpVerts, m_highLightColor, m_edgeWidthGL, m_renderer->GetCamera());
}













//  +--------------------------------------------------------------------------------------------+
//  |																							 |
//  |			HostScene management     														 |
//  |																							 |
//  +--------------------------------------------------------------------------------------------+

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::AddNavMeshToScene                                           |
//  |  Adds all navmesh assets to the scene.                                LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::AddNavMeshToScene()
{
	printf("Adding NavMesh assets to scene...\n");
	Timer timer;
	AddPolysToScene();
	AddVertsToScene();
	AddEdgesToScene();
	printf("NavMesh assets added in %.3fms\n", timer.elapsed());
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::RemoveNavMeshFromScene                                      |
//  |  Removes all navmesh assets from the scene.                           LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::RemoveNavMeshFromScene()
{
	RemovePolysFromScene();
	RemoveVertsFromScene();
	RemoveEdgesFromScene();
	RemoveTmpOMCs();
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::AddPolysToScene                                             |
//  |  Adds the navmesh polygons to the scene as a single mesh.             LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::AddPolysToScene()
{
	m_polyMeshID = m_renderer->AddMesh(m_meshFileName.c_str(), m_dir, 1.0f);
	HostScene::meshPool[m_polyMeshID]->name = "NavMesh";
	m_polyInstID = m_renderer->AddInstance(m_polyMeshID, mat4::Identity());
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::RemovePolysFromScene                                        |
//  |  Removes the navmesh polygons from the scene.                         LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::RemovePolysFromScene()
{
	if (m_polyInstID >= 0) m_renderer->RemoveNode(m_polyInstID);
	m_polyInstID = -1;
	// TODO: Remove the old navmesh mesh to prevent memory leaks
	//if (m_navmeshMeshID >= 0) m_renderer->RemoveMesh(m_navmeshMeshID);
	//m_navmeshMeshID = -1;
	m_renderer->SynchronizeSceneData();
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::AddVertsToScene                                             |
//  |  Add the navmesh vertices to the scene.                               LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::AddVertsToScene()
{
	mat4 scale = mat4::Scale(m_vertWidth);
	for (std::vector<Vert>::iterator i = m_verts.begin(); i != m_verts.end(); i++)
	{
		// Checking if vert is part of an unidirectional OMC
		bool addArrowCone = false;
		if (isOffMeshVert(i->idx)) for (auto p = m_polys.rbegin(); p != m_polys.rend(); p++)
			if (p->ref == i->polys[0] && !p->poly && p->omc)
				{
					if (!(p->omc->flags & DT_OFFMESH_CON_BIDIR)) addArrowCone = true;
					scale = mat4::Scale(p->omc->rad); // Retrieve OMC end-point radius
					break;
				}

		// If part of an unidirectional OMC, find the edge
		if (addArrowCone) for (auto e = m_edges.begin(); e != m_edges.end(); e++)
		{
			if (e->v1 == i->idx) // i is the start of an OMC
			{
				addArrowCone = false;
				break;
			}
			if (e->v2 == i->idx) // i is the end of an OMC
			{
				// calculate the edge angle to find the arrow cone transform
				float3 v1 = *m_verts[e->v1].pos, v2 = *m_verts[e->v2].pos;
				float3 v1v2 = normalize(v2 - v1);
				mat4 sca = mat4::Scale(make_float3(m_arrowWidth, m_arrowHeight, m_arrowWidth));
				mat4 rot = mat4::Rotate(normalize(cross({ 0, 1, 0 }, v1v2)), -acosf(v1v2.y));
				mat4 tra = mat4::Translate(v2 - v1v2 * m_arrowHeight * .5f);
				i->instID = m_renderer->AddInstance(m_arrowMeshID, tra * rot * sca);
				addArrowCone = true;
				break;
			}
		}

		// add a normal vertex
		if (!addArrowCone && !i->polys.empty())
			i->instID = m_renderer->AddInstance(m_vertMeshID, mat4::Translate(*i->pos) * scale);
	}
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::RemoveVertsFromScene                                        |
//  |  Removes all vertex instances from the scene.                         LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::RemoveVertsFromScene()
{
	for (std::vector<Vert>::iterator i = m_verts.begin(); i != m_verts.end(); i++)
	{
		if (i->instID >= 0) m_renderer->RemoveNode(i->instID);
		i->instID = -1;
	}
	m_renderer->SynchronizeSceneData();
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::AddEdgesToScene                                             |
//  |  Adds the navmesh edges to the scene.                                 LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::AddEdgesToScene()
{
	for (std::vector<Edge>::iterator i = m_edges.begin(); i != m_edges.end(); i++)
	{
		float3 v1 = *m_verts[i->v1].pos, v2 = *m_verts[i->v2].pos;
		float3 v1v2 = v2 - v1; float len = length(v1v2); v1v2 /= len;
		mat4 sca = mat4::Scale(make_float3(m_edgeWidth, len - m_vertWidth, m_edgeWidth));
		mat4 rot = mat4::Rotate(normalize(cross({ 0, 1, 0 }, v1v2)), -acosf(v1v2.y));
		mat4 tra = mat4::Translate((v1 + v2) / 2);
		i->instID = m_renderer->AddInstance(m_edgeMeshID, tra * rot * sca);
	}
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::RemoveEdgesFromScene                                        |
//  |  Removes all edge instances from the scene.                           LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::RemoveEdgesFromScene()
{
	for (std::vector<Edge>::iterator i = m_edges.begin(); i != m_edges.end(); i++)
	{
		if (i->instID >= 0) m_renderer->RemoveNode(i->instID);
		i->instID = -1;
	}
	m_renderer->SynchronizeSceneData();
}













//  +--------------------------------------------------------------------------------------------+
//  |																							 |
//  |			Agents																			 |
//  |																							 |
//  +--------------------------------------------------------------------------------------------+

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::UpdateAgentPositions                                        |
//  |  Informs the renderer of the agent's new positions.                   LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::UpdateAgentPositions()
{
	for (std::vector<ShaderAgent>::iterator it = m_agents.begin(); it != m_agents.end(); it++)
		m_renderer->SetNodeTransform(it->instID, it->agent->GetTransform());
	m_renderer->SynchronizeSceneData();
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::AddAgentToScene                                             |
//  |  Places an agent at the given position.                               LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::AddAgentToScene(Agent* agent)
{
	int instID = m_renderer->AddInstance(m_agentMeshID, agent->GetTransform());
	m_agents.push_back({ instID, agent });
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::RemoveAgent                                                 |
//  |  Removes an individual agent from the scene.                          LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::RemoveAgentFromScene(Agent* agent)
{
	for (std::vector<ShaderAgent>::iterator i = m_agents.begin(); i != m_agents.end(); i++)
		if (i->agent == agent)
		{
			if (i->instID >= 0) m_renderer->RemoveNode(i->instID);
			m_renderer->SynchronizeSceneData();
			i->instID = -1;
			m_agents.erase(i);
			return;
		}
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::RemoveAllAgents                                             |
//  |  Removes all agents from the scene.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::RemoveAllAgents()
{
	for (std::vector<ShaderAgent>::iterator i = m_agents.begin(); i != m_agents.end(); i++)
	{
		if (i->instID >= 0) m_renderer->RemoveNode(i->instID);
		i->instID = -1;
	}
	m_agents.clear();
	m_renderer->SynchronizeSceneData();
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::SetExcludedPolygons                                         |
//  |  Sets the polygons that are excluded by the agent's filter.           LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::SetExcludedPolygons(const NavMeshNavigator* navmesh, short flags)
{
	m_excludedPolygons.clear();
	if (!navmesh) return;
	for (int i = 0; i < navmesh->GetDetourMesh()->getMaxTiles(); i++)
	{
		const dtMeshTile* tile = navmesh->GetDetourMesh()->getTile(i);
		if (!tile->header) return; // invalid tile
		for (int j = 0; j < tile->header->polyCount; j++)
		{
			const dtPoly* poly = &tile->polys[j];
			if (poly->flags & flags || !poly->flags) m_excludedPolygons.push_back(poly);
		}
	}
}













//  +--------------------------------------------------------------------------------------------+
//  |																							 |
//  |			Selecting																		 |
//  |																							 |
//  +--------------------------------------------------------------------------------------------+

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::SelectPoly                                                  |
//  |  Highlights a polygon given a triangle ID of the navmesh instance.          |
//  |  This method assumes that the correct instance ID has been selected.        |
//  |  Returns a const pointer to the original of the selected poly.        LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshShader::Poly* NavMeshShader::SelectPoly(int triangleID)
{
	Deselect();
	for (int i = 0; i < (int)m_polys.size(); i++)
		for (int t : m_polys[i].triIDs) if (t == triangleID)
			return m_polySelect = &m_polys[i];
	return 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::SelectVert                                                  |
//  |  Hightlights a vertex instance.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshShader::Vert* NavMeshShader::SelectVert(int instanceID)
{
	Deselect();
	if (instanceID < 0) return 0;
	for (size_t i = 0; i < m_verts.size(); i++)
		if (m_verts[i].instID == instanceID) return m_vertSelect = &m_verts[i];
	return 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::SelectEdge                                                  |
//  |  Hightlights an edge instance.                                        LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshShader::Edge* NavMeshShader::SelectEdge(int instanceID)
{
	Deselect();
	if (instanceID < 0) return 0;
	for (size_t i = 0; i < m_edges.size(); i++)
		if (m_edges[i].instID == instanceID) return m_edgeSelect = &m_edges[i];
	return 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::SelectAgent                                                 |
//  |  Hightlights an agent instance.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
Agent* NavMeshShader::SelectAgent(int instanceID)
{
	Deselect();
	if (instanceID < 0) return 0;
	for (size_t i = 0; i < m_agents.size(); i++)
		if (m_agents[i].instID == instanceID)
			return (m_agentSelect = new ShaderAgent(m_agents[i]))->agent;
	return 0;
}













//  +--------------------------------------------------------------------------------------------+
//  |																							 |
//  |			Editing																			 |
//  |																							 |
//  +--------------------------------------------------------------------------------------------+

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::SetTmpVert                                                  |
//  |  Adds a temporary vertex to the scene.                                LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::SetTmpVert(float3 pos)
{
	RemoveTmpVert();
	m_tmpVert.pos = new float3(pos); // Vert struct needs a pointer
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::RemoveTMPVert                                               |
//  |  Removes the temporary vertex from the scene.                         LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::RemoveTmpVert()
{
	delete m_tmpVert.pos; // locally owned
	m_tmpVert.pos = 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::AddTmpOMC                                                   |
//  |  Adds a temporary off-mesh connection during runtime.                 LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::AddTmpOMC(float3 v0, float3 v1)
{
	m_tmpEdges.push_back({ (int)m_tmpVerts.size(), (int)m_tmpVerts.size() + 1 });
	m_tmpVerts.push_back({ new float3(v0), (int)m_tmpVerts.size() });
	m_tmpVerts.push_back({ new float3(v1), (int)m_tmpVerts.size() });
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::RemoveTmpOMCs                                               |
//  |  Removes all temporary off-mesh connections from the scene.           LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::RemoveTmpOMCs()
{
	for (Vert v : m_tmpVerts) delete v.pos; // locally owned
	m_tmpVerts.clear();
	m_tmpEdges.clear();
}













//  +--------------------------------------------------------------------------------------------+
//  |																							 |
//  |			Internal Representation															 |
//  |																							 |
//  +--------------------------------------------------------------------------------------------+

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::UpdateMesh                                                  |
//  |  Builds a new internal representation based on a new navmesh.         LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::UpdateMesh(NavMeshNavigator* navmesh)
{
	Clean();
	m_meshFileName = ".tmp." + std::string(navmesh->GetID()) + ".obj";
	ExtractObjects(navmesh->GetDetourMesh());
	SaveAsMesh(navmesh);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::AddEdgeToEdgesAndPreventDuplicates                          |
//  |  Helper function for NavMeshShader::ExtractVertsAndEdges.             LH2'19|
//  +-----------------------------------------------------------------------------+
void AddEdgeToEdgesAndPreventDuplicates(std::vector<NavMeshShader::Edge>& edges, int v1, int v2, dtPolyRef poly)
{
	for (std::vector<NavMeshShader::Edge>::iterator i = edges.begin(); i != edges.end(); i++)
		if ((i->v1 == v1 && i->v2 == v2) || (i->v1 == v2 && i->v2 == v1))
		{
			i->poly2 = poly; // edge already exists, so add the second poly
			return;
		}
	edges.push_back({ v1, v2, (int)edges.size(), -1, poly }); // add a new edge
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::ExtractObjects                                              |
//  |  Creates an internal representation of verts, edges, and polys.             |
//  |  For every tile, it														  |
//  |  1) instantiates all Vert instances with a pointer to the corresponding     |
//  |     vertex position in the dtNavMesh.                                       |
//  |  2) instantiates all Poly instances with a pointer to the corresponding     |
//  |     dtPoly in the dtNavMesh, and a list of triangles indices that           |
//  |     correspond to this polygon. These indices are determined by counting    |
//  |     the number of triangles that have been assigned, and therefore assumes  |
//  |     that they were read in the same order as in `WriteTileToMesh`. This is  |
//  |     a result of separating the two functionalities for readability, and     |
//  |     making sure that the polygons are added before the OMCs.                |
//  |  3) hands each Vert a list of dtPolyRefs (IDs) that they're in              |
//  |  4) instantiates all Edge instances with the indices of both verts,         |
//  |     as well as two dtPolyRefs to the two polygons they connect.             |
//  |  5) adds the off-mesh connections last, using the same Vert/Edge/Poly       |
//  |     struct as polygons, but using the `Poly::omc` member instead of         |
//  |     `Poly::poly`, which share a union.									  |
//  |  6) keeps a list of vertex index offsets for normal-, detail-, and OMC      |
//  |     verts for each tile.                                              LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::ExtractObjects(const dtNavMesh* mesh)
{
	// Pre-allocate memory for the verts and edges
	int totalVerts = 0;
	for (int a = 0; a < mesh->getMaxTiles(); a++) if (mesh->getTile(a)->header)
	{
		dtMeshHeader* header = mesh->getTile(a)->header;
		totalVerts += header->vertCount + header->detailVertCount + header->offMeshConCount * 2;
	}
	m_verts.reserve(totalVerts); // total amount of verts
	m_edges.reserve(totalVerts); // at least this many edges

	// Loop over tiles in the navmesh
	int tileBaseIdx = 0, triCount = 0, nVerts, nDetail, nOMC;
	for (int a = 0; a < mesh->getMaxTiles(); a++) if (mesh->getTile(a)->header)
	{
		const dtMeshTile* tile = mesh->getTile(a);
		nVerts = tile->header->vertCount;
		nDetail = tile->header->detailVertCount;
		nOMC = tile->header->offMeshConCount;

		// Adding Verts and their positions
		for (int i = 0; i < nVerts; ++i)
		{
			const float* v = &tile->verts[i * 3];
			m_verts.push_back({ (float3*)v, tileBaseIdx + i });
		}
		for (int i = 0; i < nDetail; ++i)
		{
			const float* v = &tile->detailVerts[i * 3];
			m_verts.push_back({ (float3*)v, tileBaseIdx + nVerts + i });
		}

		// Adding Vert polygon associations, Edges, and Polys
		dtPolyRef refBase = mesh->getPolyRefBase(tile);
		for (int b = 0; b < tile->header->polyCount; b++)
		{
			const dtPoly* poly = &tile->polys[b];
			dtPolyRef ref = refBase + b;
			if (poly->getType() == DT_POLYTYPE_OFFMESH_CONNECTION) continue;

			// Adding poly and collecting triangle IDs
			m_polys.push_back(Poly{ poly, ref });
			for (int c = 0; c < tile->detailMeshes[b].triCount; c++)
				m_polys.back().triIDs.push_back(triCount++); // NOTE: depends on .obj file writing of faces

			for (int c = 0; c < poly->vertCount; c++) // for every poly vert
			{
				m_verts[tileBaseIdx + poly->verts[c]].polys.push_back(ref);
				if (c < poly->vertCount-1) // adding the first n-1 edges
					AddEdgeToEdgesAndPreventDuplicates(m_edges, tileBaseIdx + poly->verts[c], tileBaseIdx + poly->verts[c + 1], ref);
				else // adding the last edge, connecting the last- and first vertex
					AddEdgeToEdgesAndPreventDuplicates(m_edges, tileBaseIdx + poly->verts[c], tileBaseIdx + poly->verts[0], ref);
			}
		}

		// Adding off-mesh connections
		Poly omcPoly;
		int v1, v2;
		for (int j = 0; j < nOMC; j++)
		{
			const dtOffMeshConnection* omc = &tile->offMeshCons[j];
			omcPoly.omc = omc;
			omcPoly.ref = refBase + omc->poly;
			v1 = tileBaseIdx + nVerts + nDetail + (j * 2);
			v2 = tileBaseIdx + nVerts + nDetail + (j * 2) + 1;
			m_verts.push_back({ (float3*)omc->pos, v1 });
			m_verts.back().polys.push_back(omcPoly.ref);
			m_verts.push_back({ (float3*)(omc->pos+3), v2 });
			m_verts.back().polys.push_back(omcPoly.ref);
			m_edges.push_back({ v1, v2, (int)m_edges.size(), -1, omcPoly.ref });
			m_polys.push_back(omcPoly);
		}

		// Keeping track of the vertex offsets
		m_vertOffsets.push_back(int3{ tileBaseIdx, tileBaseIdx + nVerts, tileBaseIdx + nVerts + nDetail });
		tileBaseIdx += nVerts + nDetail + nOMC * 2;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::WriteTileToMesh                                             |
//  |  Writes the given navmesh tile to the given opened file.              LH2'19|
//  +-----------------------------------------------------------------------------+
void WriteTileToMesh(const dtMeshTile* tile, FILE* f)
{
	// Writing vertices
	int vertCount = 0;
	for (int i = 0; i < tile->header->vertCount; ++i)
	{
		const float* v = &tile->verts[i * 3];
		fprintf(f, "v %.5f %.5f %.5f\n", v[0], v[1], v[2]);
		vertCount++;
	}
	for (int i = 0; i < tile->header->detailVertCount; ++i)
	{
		const float* v = &tile->detailVerts[i * 3];
		fprintf(f, "v %.5f %.5f %.5f\n", v[0], v[1], v[2]);
		vertCount++;
	}
	fprintf(f, "# %i vertices\n\n", vertCount);

	// Writing texture coordinates
	fprintf(f, "vt 0 0\n");
	fprintf(f, "vt 0 1\n");
	fprintf(f, "vt 1 1\n");
	fprintf(f, "# 3 texture vertices\n\n");

	// Writing normals
	int normCount = 0;
	for (int i = 0; i < tile->header->polyCount; ++i)
	{
		const dtPoly poly = tile->polys[i];
		if (poly.getType() == DT_POLYTYPE_OFFMESH_CONNECTION)
			continue;
		const dtPolyDetail pd = tile->detailMeshes[i];

		// For each triangle in the polygon
		for (int j = 0; j < pd.triCount; ++j)
		{
			const unsigned char* tri = &tile->detailTris[(pd.triBase + j) * 4];

			// Find the three vertex pointers
			const float* v[3];
			for (int k = 0; k < 3; ++k)
			{
				if (tri[k] < poly.vertCount)
					v[k] = &tile->verts[poly.verts[tri[k]] * 3];
				else
					v[k] = &tile->detailVerts[(pd.vertBase + tri[k] - poly.vertCount) * 3];
			}

			// Calculate the normal
			float3 v0 = make_float3(v[0][0], v[0][1], v[0][2]);
			float3 v1 = make_float3(v[1][0], v[1][1], v[1][2]);
			float3 v2 = make_float3(v[2][0], v[2][1], v[2][2]);
			float3 n = cross(v1 - v0, v2 - v0);
			normalize(n);
			if (n.y < 0) n = -n; // ensures all normals point up

			// Write the normal to the file
			fprintf(f, "vn %.5f %.5f %.5f\n", n.x, n.y, n.z);
			normCount++;
		}
	}
	fprintf(f, "# %i normals\n\n", normCount);

	// Writing faces
	int faceCount = 0;
	fprintf(f, "usemtl navmesh\n");
	for (int i = 0; i < tile->header->polyCount; ++i)
	{
		const dtPoly poly = tile->polys[i];

		// If it's an off-mesh connection, continue
		if (poly.getType() == DT_POLYTYPE_OFFMESH_CONNECTION) continue;

		// For each triangle in the polygon
		const dtPolyDetail pd = tile->detailMeshes[i];
		for (int j = 0; j < pd.triCount; ++j)
		{
			// triangle vertices (tri[n]) are indices of poly.verts, poly.verts holds the actual indices
			const unsigned char* tri = &tile->detailTris[(pd.triBase + j) * 4];

			// Find the three vertex indices
			int v[3];
			for (int k = 0; k < 3; ++k)
			{
				if (tri[k] < poly.vertCount) v[k] = poly.verts[tri[k]];
				else // poly.verts indices beyond the vert count refer to the detail verts
					v[k] = tile->header->vertCount + pd.vertBase + (tri[k] - poly.vertCount);
			}

			// Write the face to the file
			fprintf(f, "f");
			for (int k = 0; k < 3; k++)
				fprintf(f, " %i/%i/%i", v[k] + 1, k + 1, faceCount + 1); // +1 because .obj indices start at 1
			fprintf(f, "\n");

			faceCount++;
		}
	}
	fprintf(f, "# %i faces\n\n", faceCount);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::SaveAsMesh                                                  |
//  |  Saves the navmesh as an .obj file.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::SaveAsMesh(NavMeshNavigator* navmesh)
{
	// Opening file
	Timer timer;
	const char* ID = navmesh->GetID();
	std::string filename = m_dir + m_meshFileName;
	printf("Saving navmesh as wavefront in '%s'... ", filename.c_str());
	FILE* f;
	fopen_s(&f, filename.c_str(), "w");

	// Error handling
	if (!f)
	{
		printf("ERROR: File '%s' could not be opened\n", filename.c_str());
		return;
	}
	const dtNavMesh* mesh = navmesh->GetDetourMesh();
	if (!mesh)
	{
		printf("ERROR: navmesh '%s' is null\n", ID);
		return;
	}

	// Writing header
	fprintf(f, "#\n# Wavefront OBJ file\n");
	fprintf(f, "# Navigation mesh\n# ID: '%s'\n", ID);
	fprintf(f, "# Automatically generated by 'recastnavigation.cpp'\n");
	fprintf(f, "#\nmtllib %s\n\n", m_matFileName.c_str());

	// Writing one group per tile
	for (int i = 0; i < mesh->getMaxTiles(); ++i) if (mesh->getTile(i)->header)
	{
		fprintf(f, "g Tile%3i\n", i);
		WriteTileToMesh(mesh->getTile(i), f);
	}

	fclose(f);
	printf("%.3fms\n", timer.elapsed());
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader::Clean                                                       |
//  |  Cleans the internal representation and removes assets from the scene.      |
//  |																		LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshShader::Clean()
{
	RemoveNavMeshFromScene();
	RemoveNavMeshFromGL();
	RemoveAllAgents();
	RemoveTmpVert();
	RemoveTmpOMCs();

	Deselect();
	m_verts.clear();
	m_edges.clear();
	m_polys.clear();

	m_path = 0;
	m_pathStart = m_pathEnd = 0;

	RemoveFile((m_dir + m_meshFileName).c_str());
	m_meshFileName = "";
}

} // namespace lighthouse2

// EOF