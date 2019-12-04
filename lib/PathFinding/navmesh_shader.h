/* navmesh_shader.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


   The NavMeshShader provides a visual representation of the navmesh
   and its functionality. An example of a typical use case:

	1) Construct an instance with a pointer to the renderer, and a
	  directory containing the assets found in (lib/pathfinding/assets)
		-> `NavMeshShader shader(renderer, directory)`

	2) Give it a NavMeshNavigator -> `shader.UpdateMesh(navmesh)`

	3) Call either `AddNavMeshToScene()` (recommended) or `AddNavMeshToGL()`

	4) Call `DrawGL()` after rendering to do all OpenGL functionality

	5) Call `UpdateAgentPositions()` after the physics update
		to inform the renderer of the new agent positions

*/

#pragma once

#include "DetourNavMesh.h"

#include "rendersystem.h"		// RenderAPI
#include "navmesh_navigator.h"  // NavMeshNavigator
#include "navmesh_agents.h"		// Agent

#define NAVMESH_VERTEX_MESH_FILE "vertex.obj"
#define NAVMESH_EDGE_MESH_FILE "edge.obj"
#define NAVMESH_ARROW_MESH_FILE "arrowcone.obj"
#define NAVMESH_AGENT_MESH_FILE "agent.obj"

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  NavMeshShader                                                              |
//  |  NavMeshShader class handles the visualization of the navmesh.        LH2'19|
//  +-----------------------------------------------------------------------------+
class NavMeshShader
{
public:
	NavMeshShader(RenderAPI* renderer, const char* dir)
		: m_renderer(renderer), m_dir(dir)
	{
		// Initialize meshes
		m_vertMeshID = m_renderer->AddMesh(NAVMESH_VERTEX_MESH_FILE, m_dir, 1.0f);
		m_edgeMeshID = m_renderer->AddMesh(NAVMESH_EDGE_MESH_FILE, m_dir, 1.0f);
		m_arrowMeshID = m_renderer->AddMesh(NAVMESH_ARROW_MESH_FILE, m_dir, 1.0f);
		m_agentMeshID = m_renderer->AddMesh(NAVMESH_AGENT_MESH_FILE, m_dir, 1.0f);
		HostScene::meshPool[m_vertMeshID]->name = "navmesh_vertex";
		HostScene::meshPool[m_edgeMeshID]->name = "navmesh_edge";
		HostScene::meshPool[m_arrowMeshID]->name = "navmesh_arrowcone";
		HostScene::meshPool[m_agentMeshID]->name = "navmesh_agent";
	};
	~NavMeshShader() {};

	// GL Drawing
	void DrawGL() const;
	void AddNavMeshToGL() { m_shadePolys = m_shadeVerts = m_shadeEdges = true; };
	void AddPolysToGL() { m_shadePolys = true; };
	void AddVertsToGL() { m_shadeVerts = true; };
	void AddEdgesToGL() { m_shadeEdges = true; };
	void RemoveNavMeshFromGL() { m_shadePolys = m_shadeVerts = m_shadeEdges = false; };
	void RemovePolysFromGL() { m_shadePolys = false; };
	void RemoveVertsFromGL() { m_shadeVerts = false; };
	void RemoveEdgesFromGL() { m_shadeEdges = false; };

	// HostScene Management
	void AddNavMeshToScene();
	void AddPolysToScene();
	void AddVertsToScene();
	void AddEdgesToScene();
	void RemoveNavMeshFromScene();
	void RemovePolysFromScene();
	void RemoveVertsFromScene();
	void RemoveEdgesFromScene();

	// Agents & Paths
	void UpdateAgentPositions();
	void AddAgentToScene(Agent* agent);
	void RemoveAgentFromScene(Agent* agent);
	void RemoveAllAgents();
	void SetExcludedPolygons(const NavMeshNavigator* navmesh, short flags);
	void SetPath(const std::vector<NavMeshNavigator::PathNode>* path) { m_path = path; };
	void SetPathStart(const float3* start) { m_pathStart = start; };
	void SetPathEnd(const float3* end) { m_pathEnd = end; };

	// Internal Representation
	void UpdateMesh(NavMeshNavigator* navmesh);
	void Clean();
	struct Poly { const dtPoly* poly = 0; dtPolyRef ref = 0; std::vector<int> triIDs; const dtOffMeshConnection* omc = 0; };
	struct Vert { float3* pos = 0; int idx = -1, instID = -1; std::vector<dtPolyRef> polys; };
	struct Edge { int v1 = -1, v2 = -1, idx = -1, instID = -1; dtPolyRef poly1 = 0, poly2 = 0; int arrowInstID = -1; };
	//struct OMC { const dtOffMeshConnection* omc = 0; int v1InstID = -1, v2InstID = -1, edgeInstID = -1; };
	bool isNormalVert(int idx) const { for (auto i : m_vertOffsets) { if (idx >= i.x && idx < i.y) return true; } return false; };
	bool isDetailVert(int idx) const { for (auto i : m_vertOffsets) { if (idx >= i.y && idx < i.z) return true; } return false; };
	bool isOffMeshVert(int idx) const { if (isNormalVert(idx)) { return false; } if (isDetailVert(idx)) { return false; } return true; };
	inline int GetVertIdx(int instID) { for (auto v : m_verts) if (v.instID == instID) return v.idx; return -1; };
	inline float3* GetVertPos(int idx) { return m_verts[idx].pos; };

	// Object Selecting
	void Deselect() { m_vertSelect = 0; m_edgeSelect = 0; m_polySelect = 0; if (m_agentSelect) delete m_agentSelect; m_agentSelect = 0; };
	Poly* SelectPoly(int triangleID);
	Vert* SelectVert(int instanceID);
	Edge* SelectEdge(int instanceID);
	Agent* SelectAgent(int instanceID);
	bool isAgent(int meshID) const { return meshID == m_agentMeshID; };
	bool isPoly(int meshID) const { return meshID == m_polyMeshID; };
	bool isVert(int meshID) const { return (meshID == m_vertMeshID || meshID == m_arrowMeshID); };
	bool isEdge(int meshID) const { return meshID == m_edgeMeshID; };

	// Editing
	void SetTmpVert(float3 pos);
	void RemoveTmpVert();
	void AddTmpOMC(float3 v0, float3 v1);
	void RemoveTmpOMCs();

private:
	RenderAPI* m_renderer;
	const char* m_dir;

	// GL Drawing
	bool m_shadePolys = false, m_shadeVerts = false, m_shadeEdges = false;
	const float m_edgeWidthGL = 5.0f, m_vertWidthGL = 20.0f;   // width of the GL shaded lines (in pixels)
	const float m_pathWidth = 3.0f, m_beaconWidth = 10.0f;	   // width of the GL shaded lines (in pixels)
	const float m_agentImpulseScale = 5.0f;					   // scale of the GL shaded agent impulse
	const float3 m_beaconLen = make_float3(0.0f, 4.0f, 0.0f);  // length of the GL shaded path beacons (in pixels)
	const float4 m_polyColor = { 0, 1.0f, 1.0f, 0.2f };	 // color of the GL shaded polys (rgba)
	const float4 m_vertColor = { 1.0f, 0, 1.0f, 0.2f };	 // color of the GL shaded verts (rgba)
	const float4 m_edgeColor = { 1.0f, 0, 1.0f, 0.2f };	 // color of the GL shaded edges (rgba)
	const float4 m_excludedColor = { 1.0f, 0, 0, .5f };  // color of the excluded polygons (rgba)
	const float4 m_pathColor = { 1.0f, 0.0f, 0.0f, 0.5f };	// color of the GL shaded path (rgba)
	const float4 m_beaconStColor = { 0, 1.0f, 0, 1.0f };	// color of the GL shaded path start beacon (rgba)
	const float4 m_beaconEnColor = { 1.0f, 0, 0, 1.0f };	// color of the GL shaded path end beacon (rgba)
	const float4 m_highLightColor = { 1.0f, 1.0f, 0.0f, .5f };	// color of highlighted navmesh objects (rgba)
	void DrawAgentHighlightGL() const;
	void DrawAgentImpulse() const;
	void PlotPath() const;
	void DrawPathMarkers() const;
	void DrawTmpOMCs() const;

	// HostScene Management
	int m_polyMeshID = -1, m_polyInstID = -1;
	int m_vertMeshID = -1, m_edgeMeshID = -1, m_agentMeshID = -1, m_arrowMeshID = -1;
	std::string m_meshFileName;
	const std::string m_matFileName = "navmesh.mtl";
	const float m_vertWidth = .3f, m_edgeWidth = .1f; // in world coordinates
	const float m_arrowWidth = .4f, m_arrowHeight = .7f; // in world coordinates
	void SaveAsMesh(NavMeshNavigator* navmesh);

	// Agents & Paths
	struct ShaderAgent { int instID = -1; Agent* agent; };
	std::vector<ShaderAgent> m_agents;
	std::vector<const dtPoly*> m_excludedPolygons; // polygons to be highlighted as 'excluded'
	const std::vector<NavMeshNavigator::PathNode>* m_path = 0;
	const float3 *m_pathStart = 0, *m_pathEnd = 0; // draws a path beacon when set
	
	// Internal Representation
	std::vector<Vert> m_verts; // (verts, detailVerts, offMeshVerts) * nTiles
	std::vector<Edge> m_edges; // (edges, offMeshEdges) * nTiles
	std::vector<Poly> m_polys; // (polys, offMeshConnections) * nTiles
	std::vector<Vert> m_tmpVerts; // used for navmesh pruning and adding OMCs
	std::vector<Edge> m_tmpEdges; // used for navmesh pruning and adding OMCs
	std::vector<int3> m_vertOffsets; // idx offset of (poly verts, detail verts, omc verts) per tile
	void ExtractObjects(const dtNavMesh* navmesh);

	// Object Selecting
	Poly* m_polySelect = 0;
	Vert* m_vertSelect = 0;
	Edge* m_edgeSelect = 0;
	ShaderAgent* m_agentSelect = 0; // points to a copy of the ShaderAgent

	// Editing
	Vert m_tmpVert;
};

} // namespace lighthouse2

// EOF