# Pathfinding

The pathfinding module is a wrapper for *Recast* and *Detour*, which respectively provide the navmesh generation and path planning. Since these libraries are git submodules, please don't forget to `git submodule init` and `git submodule update` after pulling.

**Navigation Meshes**  
A NavMesh, although called a mesh, is not meant to be shaded. It is a simplified version of the scene that tells NPCs where they can and cannot go. You can think of it as a collection of (sufficiently) level surfaces and how they are connected. Navmesh Generation is the process of automatically creating such a navmesh for any given scene.  
A navmesh contains a polygon mesh, a detail mesh, and off-mesh connections. The polygon mesh is the crude representation of the scene and the connections between polygons. The detail mesh gives more detail about the height at each point of the polygon mesh. An off-mesh connection is a manually added edge connection between two points on the navmesh, which can either be bidirectional or unidirectional. They are typically added to represent ziplines, jumping down from platforms, narrow passages, or other two-dimensional connections that use a special animation.

**Major Classes**  
The major classes in this module are the `NavMeshBuilder`, the `NavMeshNavigator`, the `Agent`, and the `NavMeshShader`. The builder handles navmesh generation, the navigator represents the navmesh during runtime, agents use this navmesh to move a rigid body around, and the shader integrates with the Lighthouse2 `RenderAPI` to give a visual representation of the navmesh for debug purposes.

Some common Detour classes the user might interact with:
* `dtNavMesh`: A finalized navmesh that can be used by Detour. This is the core of a `NavMeshNavigator`.
* `dtNavMeshQuery`: A class that can perform queries on the `dtNavMesh` (e.g. pathfinding). This is also a member of the `NavMeshNavigator`.
* `dtQueryFilter`: A list of polygon flags the agent can/cannot traverse, and a list of its traversal cost per area type. Every `Agent` has its own filter.

## NavMesh Common
**Error Handling**  
Errors are thrown by calling `NavMeshError` with a `NavMeshStatus` and an error message (both found in `navmesh_common.h`). The error itself does not interrupt execution, but simply prints the error message. Logging is therefore done by calling this function with a `NavMeshStatus::SUCCESS` status. This way, all printing in the project (except for `NavMeshShader`) is routed through this one function. All methods prone to errors will return a `NavMeshStatus`, which can be checked for errors using `NavMeshStatus::Success` or `NavMeshStatus::Failed`.

**Flags**  
Polygons in the navmesh can be labeled with up to 16 user-defined flags. The `NavMeshFlagMapping` struct contains the mapping between labels and the actual bitwise flag. The labels can be added and removed from the mapping, and the flags can be retrieved using the label string and the `[]` operator (e.g. `flagMapping["water"]`).

**Area Types**  
Polygons can also have an area type. Similar to the flags, the area types and their labels are handles by the `NavMeshAreaMapping` struct. Labels can be add and removed, and flags can be retrieved in the same way, using the label string and the `[]` operator. Additionally, agents can have varying traversal costs for the different area. The mapping struct therefore also keeps track of the default costs per area. New agents start with these default values, and their `dtQueryFilter` can be manually adjusted in case of agent-specific costs.

**NavMeshConfig**  
These are the configurations used during navmesh generation. The `NavMeshBuilder` stores these in `NavMeshBuilder::m_config`, which are saved and loaded alongside the navmesh. Additionally, the flag- and area mappings saved by the builder are also used by the `NavMeshNavigator`. The `NavMeshConfig` struct contains the following parameters:
* `m_width`/`m_height`/`m_tileSize`/`m_borderSize`: Set by the building process; represents the voxel array dimensions
* `m_cs`/`m_ch`: The voxel cell size (width/depth) and cell height respectively
* `m_bmin`/`m_bmax`: The dimensions of the axis aligned bounding box within which the navmesh should remain
* `m_walkableSlopeAngle`: The maximum slope the agent can traverse
* `m_walkableHeight`: The minimum height undernath which the agent can walk, expressed in voxels
* `m_walkableClimb`: The maximum number of voxels the agent can climb (e.g. stairs)
* `m_walkableRadius`: The minimum width through which the agent can traverse (i.e. the agent's width)
* `m_maxEdgeLen`: The maximum length of any polygon edge.
* `m_maxSimplificationError`: The maximum distance a polygon edge can differ from that in the original contour set.
* `m_minRegionArea`: The minimum number of connected voxels that constitute a surface (filters islands)
* `m_mergeRegionArea`: The minimum number of connected voxels needed to not be merged with a neighboring surface
* `m_maxVertsPerPoly`: The maximum number of polygon vertices (3-6)
* `m_detailSampleDist`: The sampling distance used in creating the detail mesh
* `m_detailSampleMaxError`: The maximum distance the detail mesh surface can deviate from the heightfield.
* `m_partitionType`: The partitioning method to use. The options are:
    * **Watershed partitioning** (best for precomputed navmeshes and open areas)
        * the classic Recast partitioning
        * creates the nicest tessellation, but usually the slowest
        * partitions the heightfield into nice regions without holes or overlaps
        * the are some corner cases where this method produces holes and overlaps:
            * holes may appear when a small obstacle is close to large open area (triangulation won't fail)
            * overlaps may occur on narrow spiral corridors (i.e stairs) and triangulation may fail
    * **Monotone partitioning** (fast navmesh generation)
        * fastest
        * partitions the heightfield into regions without holes and overlaps (guaranteed)
        * creates long thin polygons, which sometimes causes paths with detours
    * **Layer partitioning** (best for navmeshes with small tiles)
        * quite fast
        * partitions the heighfield into non-overlapping regions
        * relies on the triangulation code to cope with holes (thus slower than monotone partitioning)
        * produces better triangles than monotone partitioning
        * does not have the corner cases of watershed partitioning
        * can be slow and create a bit ugly tessellation (still better than monotone) if you have large open areas with small obstacles (not a problem if you use tiles)
* `m_keepInterResults`: Whether or not to keep the intermediate results, such as the voxel model, pmesh, and dmesh
* `m_filterLowHangingObstacles`: Whether to filter for low hanging obstacles
* `m_filterLedgeSpans`: Whether to filter for ledge spans
* `m_filterWalkableLowHeightSpans`: Whether to filter for low height spans
* `m_printBuildStats`: Whether to print detailed build statistics
* `m_id`: A string identifying the navmesh. Used in the filename while saving. Loading a navmesh requires the user to set the ID first, as it looks for the file containing this ID.
* `m_flags`: A `NavMeshFlagMapping` instance that contains the user defined polygon flags.
* `m_areas`: A `NavMeshAreaMapping` instance that contains the user defined area types and their default traversal costs.

## NavMeshBuilder

A `NavMeshBuilder` instance is initialized with a directory in which it keeps the resulting navmeshes. A navmesh can be generated using `NavMeshBuilder::Build`, which needs a `HostScene` pointer. The resulting navmesh can be saved using `NavMeshBuilder::Serialize` and loaded with `NavMeshBuilder::Deserialize`.

**Navmesh Generation**  
The generation procedure itself consists of 5 parts.
1) **Rasterization**: the scene is converted into a voxel representation. The choice of voxel size, which can be set by the user, is a trade-off between computation time and level of detail. 
2) **Filtering**: the voxel model is filtered for walkable voxels based on the parameters set by the user.
3) **Partitioning**: placing non-overlapping 2D regions on top of these chunks.
4) **Polygon Creation**: converting these regions into connected convex polygons, represented by two meshes: the *polygon mesh* and the *detail mesh*. The polygon mesh is a crude representation of traversability and polygon connections, which is used for pathfinding. The detail mesh stores the exact surface height of each point on the polygon.
5) **Creating `dtNavMesh`**: combining these two meshes into one navmesh that can be used by Detour. When the pmesh and dmesh have been manually edited, or when off-mesh connections have been added with `NavMeshBuilder::AddOffMeshConnection`, this last step has to be redone to refresh the Detour data. Hence, editing the navmesh requires the pmesh and dmesh to still be there (see the `m_keepInterResults` configuration).

Any `HostMesh` can be prevented from influencing the navmesh generation by setting `HostMesh::excludeFromNavmesh` to true. The builder is also in charge of editing the navmesh. Polygon flags and -area types can be set with `NavMeshBuilder::SetPolyFlags` and `NavMeshBuilder::SetPolyArea` respectively, which immediately applies the changes to the current `dtNavMesh`. Off-mesh connections can be added with `NavMeshBuilder::AddOffMeshConnection`, but require a call to `NavMeshBuilder::ApplyChanges` before the changes take effect. Alternatively, these pending changes can be discarded using `NavMeshBuilder::DiscardChanges`.

If an error occurs during the generation process, the internal error status is updated there and then. Any subprocesses called after that will not commence if the error status is unsuccessful, cutting the process short. Any allocated memory is freed before returning the error status to the user.  
More info on the navmesh generation can be found in the official [Recast documentation](http://masagroup.github.io/recastdetour/group__recast.html).

## NavMeshNavigator

The `NavMeshNavigator` is the class representing a navmesh at runtime. It primarily consists of the underlying `dtNavMesh` and its `dtNavMeshQuery`.
The former of these holds all data regarding the navmesh, and the latter can be used to query it. An instance can be constructed by loading a precomputed navmesh, or directly asking one from the builder using `NavMeshBuilder::GetNavigator`.

The `NavMeshNavigator::Load` method can restore navmeshes saved by the `NavMeshBuilder`, as well as the accompanied config file for the polygon flags and area types.
The flags are user-defined qualities of a polygon that can be used to exclude certain agents from traversing them.
The area types are strings to label polygons with, and agents can have different traversal costs for each area type.
`NavMeshNavigator::GetFilter` will return a `dtQueryFilter` including/excluding the given flag labels, which are internally resolved to label indices using the loaded data (e.g. `GetFilter({"ground", "road"}, {"water"})`).
The static `s_filter`, however, is an empty filter with no labels, intended for users that don't need filtering or labeling.

The functionality of the `NavMeshNavigator` is at this point limited to path finding.
The `NavMeshNavigator::FindPath` method is meant to be an easy way of finding a path without overloading the user with arguments.
The `NavMeshNavigator::FindPathConstSize` is intended for agents with a limited fixed path size, but is also internally used by the previous method. Finding a path consists of the following stages:
 1) find the `dtPolyRef` of the polygon closest to the start/end positions
 2) if both find the same polygon, return a straight path
 3) else, ask the `dtNavMeshQuery` for a path (sequence of polygons)
 4) ask the `dtNavMeshQuery` to convert it into a smooth path (sequence of target positions)
 5) collect both the positions and polygon refs into a sequence of `NavMeshNavigator::PathNode` instances

The point of the `PathNode` struct is to prevent other calls to the navigator in between navigation updates, which leaves room for a multithreaded approach in which the update load is spread out evenly across the update interval. The pathfinding methods return a boolean indicating whether the final target appears to be reachable (true when in doubt). The path can be constrained by providing a `dtQueryFilter` instance to indicate the types of polygons that cannot be traversed by this agent (e.g. riverbed, ziplines).

Other functions include `NavMeshNavigator::FindNearestPoly`, to find the navmesh polygon closest to a given position, and `NavMeshNavigator::FindClosestPointOnPoly`, to find the point on a given poly closest to a given position. These serve little purpose to the user at this point, other than being needed by the pathfinding methods.

Since a navmesh can be used by multiple agents, the navigator does not keep internal error status like the builder. It only returns the error status as its return value.

## NavMeshAgents

A `NavMeshAgents` instance keeps track of all agents and their updates.

**Usage**  
An `Agent` can be added by calling `NavMeshAgents::AddAgent` with a `NavMeshNavigator` it should adhere to and a `RigidBody` to take control of. Note that the `Agent` owns neither its navmesh nor its rigidbody. One navmesh can be used by multiple agents of the same type, and the rigidbodies are owned and updated by the physics engine (currently `PhysicsPlaceholder`).

Agents are ideally removed by passing the `Agent` pointer to a  `NavMeshAgents::RemoveAgent` call, which notifies the `NavMeshAgents` instance of the removal. Alternatively, individual agents can be removed by calling `Agent::Kill`, which is functionally the same but doesn't notify the parent class. When the `NavMeshAgents` instance is out of agent space, it will check all agents for their alive status to see if it can overwrite any.

An agent can be given a target with `Agent::SetTarget`, which can either be a static target or, when given a pointer to a position, a dynamic target. It has two update functions: `Agent::UpdateMovement` should be called before every physics update to allow the agent to move; `Agent::UpdateNavigation` can be called less frequently, or even sporadically depending on the applicatoin, to update the agent's path. Both of these update cycles are abbreviated by the `NavMeshAgents::UpdateAgentMovement` and `NavMeshAgents::UpdateAgentBehavior` functions, the latter of which uses an internal timer to only update at a given interval, even when called prematurely. The booleans they return indicate whether any of the agent's states have actually changed (e.g. when none of the agents have a target, both functions will return false).

**Agent Steering**  
Steering behavior is performed in `Agent::UpdateMovement`. It currently includes *stop*, *seek*, and *arrival*. When an agent has no target, it comes to a halt. Otherwise, it checks if the current target position has been reached. If it has, it updates the target to the next position in the path. When the current target is known, the agent moves towards it by finding the ideal velocity vector, and modifying its current rigid body velocity to the best of its abilities (limited by speed/acceleration constraints). If it is within `m_arrival` distance of the target, the desired speed is linearly interpolated between the target and `m_arrival`.  
Steering behavior is definitely not ideal and could use improvement.

## NavMeshShader

The `NavMeshShader` provides a graphical representation of all of the above classes, intended to aid AI debugging in general, and the `ai_debugger` app in particular. An instance can be constructed by passing a directory with the object meshes (found in PathFinding/assets), and a pointer to the renderer. Please note that for this class to be interactive, the renderer will need to support scene probing, since many functions rely on instance IDs, mesh IDs, or scene positions.

**Usage**  
To shade a navmesh, pass a `NavMeshNavigator` pointer to `NavMeshShader::UpdateMesh` before calling either `NavMeshShader::AddNavMeshToScene` (recommended) or `NavMeshShader::AddNavMeshToGL` (not recommended). In order for highlights and GL shading to show, the main loop should call `NavMeshShader::DrawGL` after rendering. Additionally, when using agents, the main loop should call `NavMeshShade::UpdateAgentPositions` after the physics update to move all agents to their new location.

The shader has individual Add- and Remove functions for polygons, edges, vertices, and agents. Any of these objects can be highlighted by calling their respective `Select` functions, which return a pointer to the selected object. Call `NavMeshShader::Deselect` to remove all highlights.

Paths can be drawn by passing a pointer to a precalculated path to `NavMeshShader::SetPath`. Similarly, the path start- and end beacon can be set by passing the `float3` pointers to `NavMeshShader::SetPathStart` and `NavMeshShader::SetPathEnd`. The shader will automatically detect changes to any of the data.

To fascilitate graphical navmesh editing, the shader can add-, move-, and remove one temporary vertex with `NavMeshShader::SetTmpVert` and `NavMeshShader::RemoveTmpVert`, as well as add off-mesh connections during rutime with `NavMeshShader::AddTmpOMC`. Temporary OMCs are shaded using GL, and only become assets when the changes are applied. The temporary OMCs will then be removed from the shader, and a new navigator including the added OMCs will be added to the scene as normal.

**Internal Representation**  
The shader keeps an internal representation of the navmesh, which points to various objects within the `dtNavMesh` of the `NavMeshNavigator`.
The internal representation can be updated by calling `NavMeshShader::UpdateMesh`. This function does the following:
 1) clear the previous representation and remove all assets from the scene
 2) in ExtractObjects:
    * create `Vert` instances that point to the navmesh for their position
    * create `Poly` instances that point to the polygons in the navmesh
    * add the corresponding triangle indices to the `Poly`
    * add refs of the polygons to their `Vert` instances
    * create `Edge` instances with refs to both their associated `Vert` and `Poly` instances
    * create the `Vert`-, `Poly`, and `Edge` instances for the off-mesh connections
    * keep a list of vertex idx offsets for the normal-, detail-, and OMC verts
 3) write the navmesh as a mesh to a temporary .obj file to collectively represent the polygons

The `Vert`, `Edge`, and `Poly` instances are each stored in their `std::vector` member.
When added to the scene, the polygons will simply be read from the .obj file.
The vertices and edges are created individually and store the instanceIDs in the `Vert`/`Edge` structs.

<br/>
<br/>

## Backlog

#### NavMeshBuilder
* navmesh pruning
    * add `ChangeVert` function that changes verts in `m_pmesh` and `m_dmesh`

#### NavMeshNavigator / Agent
* BUG: when close to unreachable goal above the agent, agent goes vertically
    * It's up to the physics engine to constrain agent movement
* Don't call 'arrive' behavior on every path corner?
* Add behavior (flee/follow)


#### NavMeshShader
* make meshes transparent
    * or at least OMC vertices and polygons
* remove old navmesh mesh from render core on `RemovePolysFromScene` to save memory
* OpenGL highlights cost 10-20 fps
* BUG: OpenGL shading fails when the camera is too close (when a vertex is behind the camera) (`Camera::WorldToScreenPos` issue)