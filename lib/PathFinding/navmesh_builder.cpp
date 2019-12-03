/* navmesh_builder.cpp - Copyright 2019 Utrecht University

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

#include <vector>	// vector

#include "Recast.h"
#include "RecastDump.h"			  // duLogBuildTimes
#include "DetourNavMeshBuilder.h" // dtNavMeshCreateParams, dtCreateNavMeshData

#include "buildcontext.h"	   // BuildContext
#include "navmesh_builder.h"

#define RECAST_ERROR(X, ...) return NavMeshError(&m_status, X, "ERROR NavMeshBuilder: ", __VA_ARGS__)
#define RECAST_LOG(...) NavMeshError(0, NavMeshStatus::SUCCESS, "", __VA_ARGS__)

namespace lighthouse2
{

//  +-----------------------------------------------------------------------------+
//  |  GetMinMaxBounds                                                            |
//  |  Determines the AABB bounds of the entire input mesh.                 LH2'19|
//  +-----------------------------------------------------------------------------+
void GetMinMaxBounds( std::vector<float3>* data, float3* min, float3* max )
{
	*min = make_float3( 0.0f );
	*max = make_float3( 0.0f );
	for (std::vector<float3>::iterator it = data->begin(); it != data->end(); ++it)
	{
		if (it->x < min->x) min->x = it->x;
		if (it->y < min->y) min->y = it->y;
		if (it->z < min->z) min->z = it->z;
		if (it->x > max->x) max->x = it->x;
		if (it->y > max->y) max->y = it->y;
		if (it->z > max->z) max->z = it->z;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::NavMeshBuilder                                             |
//  |  Constructor                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshBuilder::NavMeshBuilder( const char* dir ) : m_dir( dir )
{
	m_ctx = new BuildContext();
	m_triareas = 0;
	m_heightField = 0;
	m_chf = 0;
	m_cset = 0;
	m_pmesh = 0;
	m_dmesh = 0;
	m_navMesh = 0;
	m_status = NavMeshStatus::SUCCESS;
};

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::Build                                                      |
//  |  Builds a navmesh for the given scene.                                LH2'19|
//  +-----------------------------------------------------------------------------+
NavMeshStatus NavMeshBuilder::Build()
{
	m_status = NavMeshStatus::SUCCESS;
	if (HostScene::rootNodes.empty())
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INPUT, "HostScene is nullptr\n" );

	// Extracting triangle soup
	const std::vector<HostMesh*> meshes = HostScene::meshPool;
	std::vector<HostTri> hostTris;
	std::vector<float3> vertices;
	std::vector<int3> triangles;
	int nTri = 0, instancesExcluded = 0;
	for (const HostNode* node : HostScene::nodePool) if (node && node->meshID >= 0) // for every instance
	{
		if (meshes[node->meshID]->excludeFromNavmesh) // skip if excluded
		{
			instancesExcluded++;
			continue;
		}
		hostTris = meshes[node->meshID]->triangles;
		mat4 transform = node->combinedTransform;
		for (size_t j = 0; j < hostTris.size(); j++) // for every triangle
		{
			vertices.push_back( make_float3( transform * make_float4( hostTris[j].vertex0, 1 ) ) );
			vertices.push_back( make_float3( transform * make_float4( hostTris[j].vertex1, 1 ) ) );
			vertices.push_back( make_float3( transform * make_float4( hostTris[j].vertex2, 1 ) ) );
			triangles.push_back( int3{ nTri * 3 + 0, nTri * 3 + 1, nTri * 3 + 2 } );
			nTri++;
		}
	}

	// Initializing bounds
	if (m_config.m_bmin.x == m_config.m_bmax.x ||
		m_config.m_bmin.y == m_config.m_bmax.y ||
		m_config.m_bmin.z == m_config.m_bmax.z)
		GetMinMaxBounds( &vertices, &m_config.m_bmin, &m_config.m_bmax );
	rcCalcGridSize(
		(const float*)&m_config.m_bmin,
		(const float*)&m_config.m_bmax,
		m_config.m_cs,
		&m_config.m_width,
		&m_config.m_height
	);

	// Initializing log
	if (m_config.m_printBuildStats)
	{
		RECAST_LOG( "===   Building NavMesh '%s'\n", m_config.m_id.c_str() );
		RECAST_LOG( " - Voxel grid: %d x %d cells\n", m_config.m_width, m_config.m_height );
		RECAST_LOG( " - Input mesh: %.1fK verts, %.1fK tris\n",
			vertices.size() / 1000.0f, triangles.size() / 1000.0f );
		RECAST_LOG( " - Instances excluded: %i\n", instancesExcluded );
	}
	else
		RECAST_LOG( "Building NavMesh '%s'... ", m_config.m_id.c_str() );
	if (vertices.empty())
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INPUT, "Scene is empty\n" );
	m_ctx->resetTimers();
	m_ctx->startTimer( RC_TIMER_TOTAL );

	// NavMesh generation
	RasterizePolygonSoup(
		(const int)vertices.size() * 3, (float*)vertices.data(),
		(const int)triangles.size(), (int*)triangles.data()
	);
	if (!m_config.m_keepInterResults) { delete[] m_triareas; m_triareas = 0; }
	FilterWalkableSurfaces();
	PartitionWalkableSurface();
	if (!m_config.m_keepInterResults) { rcFreeHeightField( m_heightField ); m_heightField = 0; }
	ExtractContours();
	BuildPolygonMesh();
	CreateDetailMesh();
	if (!m_config.m_keepInterResults)
	{
		rcFreeCompactHeightfield( m_chf );
		m_chf = 0;
		rcFreeContourSet( m_cset );
		m_cset = 0;
	}
	CreateDetourData();

	// Logging performance
	m_ctx->stopTimer( RC_TIMER_TOTAL );
	if (m_status.Success())
	{
		if (m_config.m_printBuildStats) // logging detailed multi-line duration log
		{
			duLogBuildTimes( *m_ctx, m_ctx->getAccumulatedTime( RC_TIMER_TOTAL ) );
			RECAST_LOG( ((BuildContext*)m_ctx)->GetBuildStats().c_str() );
		}
		else // short single-line duration log
			RECAST_LOG( "%.3fms\n", m_ctx->getAccumulatedTime( RC_TIMER_TOTAL ) / 1000.0f );
		RECAST_LOG( "   '%s' polymesh: %d vertices, %d polygons\n",
			m_config.m_id.c_str(), m_pmesh->nverts, m_pmesh->npolys );
	}

	if (m_status.Failed()) Cleanup();
	return m_status;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::RasterizePolygonSoup                                       |
//  |  Takes a triangle soup and rasterizes all walkable triangles based          |
//  |  on their slope. Results in a height map (aka voxel mold).            LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::RasterizePolygonSoup( const int vert_count, const float* verts, const int tri_count, const int* tris )
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;

	// Allocate voxel heightfield where we rasterize our input data to.
	m_heightField = rcAllocHeightfield();
	if (!m_heightField)
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::MEM, "Out of memory 'solid'\n" );

	if (!rcCreateHeightfield( m_ctx, *m_heightField, m_config.m_width, m_config.m_height,
		(const float*)&m_config.m_bmin, (const float*)&m_config.m_bmax, m_config.m_cs, m_config.m_ch ))
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not create solid heightfield\n" );

	// Allocate array that can hold triangle area types.
	// If you have multiple meshes you need to process, allocate
	// and array which can hold the max number of triangles you need to process.
	m_triareas = new unsigned char[tri_count];
	if (!m_triareas)
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::MEM, "Out of memory 'm_triareas' (%d)\n", tri_count );

	// Find triangles which are walkable based on their slope and rasterize them.
	// If your input data is multiple meshes, you can transform them here, calculate
	// the are type for each of the meshes and rasterize them.
	memset( m_triareas, 0, tri_count * sizeof( unsigned char ) );
	rcMarkWalkableTriangles( m_ctx, m_config.m_walkableSlopeAngle, verts, vert_count, tris, tri_count, m_triareas );
	if (!rcRasterizeTriangles( m_ctx, verts, vert_count, tris,
		m_triareas, tri_count, *m_heightField, m_config.m_walkableClimb ))
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not rasterize triangles\n" );

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::FilterWalkableSurfaces                                     |
//  |  Filters the correctly angled surfaces for height restrictions.       LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::FilterWalkableSurfaces()
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;

	// Once all geoemtry is rasterized, we do initial pass of filtering to
	// remove unwanted overhangs caused by the conservative rasterization
	// as well as filter spans where the character cannot possibly stand.
	if (m_config.m_filterLowHangingObstacles)
		rcFilterLowHangingWalkableObstacles( m_ctx, m_config.m_walkableClimb, *m_heightField );
	if (m_config.m_filterLedgeSpans)
		rcFilterLedgeSpans( m_ctx, m_config.m_walkableHeight, m_config.m_walkableClimb, *m_heightField );
	if (m_config.m_filterWalkableLowHeightSpans)
		rcFilterWalkableLowHeightSpans( m_ctx, m_config.m_walkableHeight, *m_heightField );

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::PartitionWalkableSurface                                   |
//  |  Transforms the heightfield into a compact height field, connects           |
//  |  neightboring walkable surfaces, erodes all surfaces by the agent           |
//  |  radius and partitions them into regions.                             LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::PartitionWalkableSurface()
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;

	// Compact the heightfield so that it is faster to handle from now on.
	// This will result more cache coherent data as well as the neighbours
	// between walkable cells will be calculated.
	m_chf = rcAllocCompactHeightfield();
	if (!m_chf)
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::MEM, "Out of memory 'chf'\n" );

	if (!rcBuildCompactHeightfield( m_ctx, m_config.m_walkableHeight, m_config.m_walkableClimb, *m_heightField, *m_chf ))
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not build compact data\n" );

	// Erode the walkable area by agent radius.
	if (!rcErodeWalkableArea( m_ctx, m_config.m_walkableRadius, *m_chf ))
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not erode\n" );

	//// (Optional) Mark areas.
	//const ConvexVolume* vols = m_geom->getConvexVolumes();
	//for (int i = 0; i < m_geom->getConvexVolumeCount(); ++i)
	//	rcMarkConvexPolyArea(m_ctx, vols[i].verts, vols[i].nverts, vols[i].hmin, vols[i].hmax, (unsigned char)vols[i].area, *m_chf);

	// Partition the heightfield so that we can use simple algorithm later to triangulate the walkable areas.
	if (m_config.m_partitionType == NavMeshConfig::SAMPLE_PARTITION_WATERSHED)
	{
		// Prepare for region partitioning, by calculating distance field along the walkable surface.
		if (!rcBuildDistanceField( m_ctx, *m_chf ))
			RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not build distance field\n" );

		// Partition the walkable surface into simple regions without holes.
		if (!rcBuildRegions( m_ctx, *m_chf, 0, m_config.m_minRegionArea, m_config.m_mergeRegionArea ))
			RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not build watershed regions\n" );
	}
	else if (m_config.m_partitionType == NavMeshConfig::SAMPLE_PARTITION_MONOTONE)
	{
		// Partition the walkable surface into simple regions without holes.
		// Monotone partitioning does not need distancefield.
		if (!rcBuildRegionsMonotone( m_ctx, *m_chf, 0, m_config.m_minRegionArea, m_config.m_mergeRegionArea ))
			RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not build monotone regions\n" );
	}
	else // SAMPLE_PARTITION_LAYERS
	{
		// Partition the walkable surface into simple regions without holes.
		if (!rcBuildLayerRegions( m_ctx, *m_chf, 0, m_config.m_minRegionArea ))
			RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not build layer regions\n" );
	}

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::ExtractContours                                            |
//  |  Extracts contours from the compact height field.                     LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::ExtractContours()
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;
	m_cset = rcAllocContourSet();
	if (!m_cset)
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::MEM, "Out of memory 'cset'\n" );

	if (!rcBuildContours( m_ctx, *m_chf, m_config.m_maxSimplificationError, m_config.m_maxEdgeLen, *m_cset ))
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not create contours\n" );

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::BuildPolygonMesh                                           |
//  |  Transforms the contours into a polygon mesh.                         LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::BuildPolygonMesh()
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;
	m_pmesh = rcAllocPolyMesh();
	if (!m_pmesh)
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::MEM, "Out of memory 'pmesh'\n" );

	if (!rcBuildPolyMesh( m_ctx, *m_cset, m_config.m_maxVertsPerPoly, *m_pmesh ))
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not triangulate contours\n" );

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::CreateDetailMesh                                           |
//  |  Creates the detailed polygon mesh.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::CreateDetailMesh()
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;
	m_dmesh = rcAllocPolyMeshDetail();
	if (!m_dmesh)
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::MEM, "Out of memory 'pmdtl'\n" );

	if (!rcBuildPolyMeshDetail( m_ctx, *m_pmesh, *m_chf, m_config.m_detailSampleDist, m_config.m_detailSampleMaxError, *m_dmesh ))
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INIT, "Could not build detail mesh\n" );

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::CreateDetourData                                           |
//  |  Creates Detour navmesh from the two poly meshes.                     LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::CreateDetourData()
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;

	if (m_pmesh->npolys < 1)
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INPUT, "Resulting NavMesh has no polygons\n" );

	// The GUI may allow more max points per polygon than Detour can handle.
	// Only build the detour navmesh if we do not exceed the limit.
	if (m_config.m_maxVertsPerPoly > DT_VERTS_PER_POLYGON)
		RECAST_ERROR( NavMeshStatus::RC | NavMeshStatus::INPUT, "MaxVertsPerPoly can't be higher than %i\n", DT_VERTS_PER_POLYGON );

	m_ctx->startTimer( RC_TIMER_TEMP );

	unsigned char* navData = 0;
	int navDataSize = 0;

	// Initialize flags and area types (on clean build only)
	if (!m_navMesh) for (int i = 0; i < m_pmesh->npolys; ++i)
	{
		m_pmesh->flags[i] = 0x1;
		m_pmesh->areas[i] = 0;
	}

	dtNavMeshCreateParams params;
	memset( &params, 0, sizeof( params ) );
	params.verts = m_pmesh->verts;
	params.vertCount = m_pmesh->nverts;
	params.polys = m_pmesh->polys;
	params.polyAreas = m_pmesh->areas;
	params.polyFlags = m_pmesh->flags;
	params.polyCount = m_pmesh->npolys;
	params.nvp = m_pmesh->nvp;
	params.detailMeshes = m_dmesh->meshes;
	params.detailVerts = m_dmesh->verts;
	params.detailVertsCount = m_dmesh->nverts;
	params.detailTris = m_dmesh->tris;
	params.detailTriCount = m_dmesh->ntris;

	// Adding off-mesh connections added during last edit
	if (!m_offMeshFlags.empty())
	{
		params.offMeshConCount = (int)m_offMeshFlags.size();
		params.offMeshConVerts = (float*)m_offMeshVerts.data();
		params.offMeshConRad = m_offMeshRadii.data();
		params.offMeshConAreas = m_offMeshAreas.data();
		params.offMeshConFlags = m_offMeshFlags.data();
		params.offMeshConUserID = m_offMeshUserIDs.data();
		params.offMeshConDir = m_offMeshDirection.data();
	}

	params.walkableHeight = m_config.m_walkableHeight;
	params.walkableRadius = m_config.m_walkableRadius;
	params.walkableClimb = m_config.m_walkableClimb;
	rcVcopy( params.bmin, m_pmesh->bmin );
	rcVcopy( params.bmax, m_pmesh->bmax );
	params.cs = m_config.m_cs;
	params.ch = m_config.m_ch;
	params.buildBvTree = true;

	if (!dtCreateNavMeshData( &params, &navData, &navDataSize ))
		RECAST_ERROR( NavMeshStatus::DT | NavMeshStatus::INIT, "Could not build Detour navmesh\n" );

	if (m_navMesh) dtFreeNavMesh( m_navMesh ); // DEBUG: does this free pmesh and dmesh?
	m_navMesh = dtAllocNavMesh();
	if (!m_navMesh)
	{
		dtFree( navData );
		RECAST_ERROR( NavMeshStatus::DT | NavMeshStatus::MEM, "Could not allocate Detour navmesh\n" );
	}

	dtStatus status;
	status = m_navMesh->init( navData, navDataSize, DT_TILE_FREE_DATA );
	if (dtStatusFailed( status ))
	{
		dtFree( navData );
		RECAST_ERROR( NavMeshStatus::DT | NavMeshStatus::INIT, "Could not init Detour navmesh\n" );
	}

	m_ctx->stopTimer( RC_TIMER_TEMP );
	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::Serialize                                                  |
//  |  Writes the navmesh to storage for future use.                        LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::Serialize( const char* dir, const char* ID )
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;

	// Saving config file
	std::string configfile = std::string( dir ) + ID + PF_NAVMESH_CONFIG_FILE_EXTENTION;
	m_config.Save( configfile.c_str() );

	// Saving dtNavMesh
	m_status = SerializeNavMesh( dir, ID, m_navMesh );
	if (m_status.Failed()) return m_status;

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::Deserialize                                                |
//  |  Reads a previously stored navmesh from storage.                      LH2'19|
//  +-----------------------------------------------------------------------------+
int NavMeshBuilder::Deserialize( const char* dir, const char* ID )
{
	if (m_status.Failed()) return NavMeshStatus::INPUT;
	Cleanup();

	// Loading config file
	std::string configfile = std::string( dir ) + ID + PF_NAVMESH_CONFIG_FILE_EXTENTION;
	m_config.Load( configfile.c_str() );

	// Loading dtNavMesh
	m_status = DeserializeNavMesh( dir, ID, m_navMesh );
	if (m_status.Failed()) return m_status;

	return NavMeshStatus::SUCCESS;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::Cleanup                                                    |
//  |  Ensures all navmesh memory allocations are deleted.                  LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshBuilder::Cleanup()
{
	if (m_triareas) delete[] m_triareas;
	m_triareas = 0;
	if (m_heightField) rcFreeHeightField( m_heightField );
	m_heightField = 0;
	if (m_chf) rcFreeCompactHeightfield( m_chf );
	m_chf = 0;
	if (m_cset) rcFreeContourSet( m_cset );
	m_cset = 0;
	if (m_pmesh) rcFreePolyMesh( m_pmesh );
	m_pmesh = 0;
	if (m_dmesh) rcFreePolyMeshDetail( m_dmesh );
	m_dmesh = 0;
	if (m_navMesh) dtFreeNavMesh( m_navMesh );
	m_navMesh = 0;

	m_offMeshVerts.clear();
	m_offMeshRadii.clear();
	m_offMeshAreas.clear();
	m_offMeshFlags.clear();
	m_offMeshUserIDs.clear();
	m_offMeshDirection.clear();
}

//  +-----------------------------------------------------------------------------+
//  |  GetOmcIndexFromPolyRef                                                     |
//  |  Finds the index of the off-mesh connection, given its dtPolyRef.           |
//  |  Returns -1 when the ref doesn't belong to an OMC.                    LH2'19|
//  +-----------------------------------------------------------------------------+
int GetOmcIndexFromPolyRef( const dtPolyRef ref, const dtNavMesh* navMesh, const std::vector<uint>& userDefinedIDs )
{
	const dtOffMeshConnection* omc = navMesh->getOffMeshConnectionByRef( ref );
	if (omc) for (int i = 0; i < (int)userDefinedIDs.size(); i++)
		if (userDefinedIDs[i] == omc->userId)
			return i;
	return -1; // No OMC found
}

//  +-----------------------------------------------------------------------------+
//  |  GetOmcIndexFromPolyRef (TODO)                                              |
//  |  Finds the index of the polygon in the dtPolyMesh, given its dtPolyRef.     |
//  |  Returns -1 when none can be found.				                    LH2'19|
//  +-----------------------------------------------------------------------------+
int GetPolyMeshIndexFromPolyRef( const dtPolyRef ref, const dtNavMesh* navMesh )
{
	//
	// TODO: convert dtPolyRef back to dtPolyMesh index
	//
	return 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::SetPolyFlags                                               |
//  |  Sets the flags of the specified polygon for both the current dtNavMesh,    |
//  |  as well as the dtPolyMesh. This way, new dtNavMesh instances will also     |
//  |  include these changes (except for clean rebuilds).                   LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshBuilder::SetPolyFlags( dtPolyRef ref, unsigned short flags )
{
	// direct/temporary saving to current dtNavMesh
	m_navMesh->setPolyFlags( ref, flags );

	// permanent saving (takes effect after ApplyChanges)
	int omcIdx = GetOmcIndexFromPolyRef( ref, m_navMesh, m_offMeshUserIDs );
	if (omcIdx > -1) // check if the ref is an off-mesh connection
	{
		m_offMeshFlags[omcIdx] = flags;
	}
	else
	{
		int pmeshIdx = GetPolyMeshIndexFromPolyRef( ref, m_navMesh );
		if (pmeshIdx > -1) m_pmesh->flags[pmeshIdx] = flags;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::SetPolyArea                                                |
//  |  Sets the area of the specified polygon for both the current dtNavMesh,     |
//  |  as well as the dtPolyMesh. This way, new dtNavMesh instances will also     |
//  |  include these changes (except for clean rebuilds).                   LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshBuilder::SetPolyArea( dtPolyRef ref, unsigned char area )
{
	// direct/temporary saving to current dtNavMesh
	m_navMesh->setPolyArea( ref, area );

	// permanent saving (takes effect after ApplyChanges)
	int omcIdx = GetOmcIndexFromPolyRef( ref, m_navMesh, m_offMeshUserIDs );
	if (omcIdx > -1) // check if the ref is an off-mesh connection
	{
		m_offMeshAreas[omcIdx] = area;
	}
	else
	{
		int pmeshIdx = GetPolyMeshIndexFromPolyRef( ref, m_navMesh );
		if (pmeshIdx > -1) m_pmesh->areas[pmeshIdx] = area;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::AddOffMeshConnection                                       |
//  |  Adds an off-mesh connection edge to the navmesh. If the connection is      |
//  |  unidirectional (e.g. ziplines, jump downs) v0 leads to v1. Requires call   |
//  |  to ApplyChanges before being effective.								LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshBuilder::AddOffMeshConnection( float3 v0, float3 v1, float radius, bool unidirectional )
{
	m_offMeshVerts.push_back( v0 );
	m_offMeshVerts.push_back( v1 );
	m_offMeshRadii.push_back( radius );
	m_offMeshAreas.push_back( 0 );
	m_offMeshFlags.push_back( 1 );
	m_offMeshUserIDs.push_back( (unsigned int)m_offMeshFlags.size() );
	m_offMeshDirection.push_back( (unidirectional ? 0 : DT_OFFMESH_CON_BIDIR) );
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::GetOmcRadius                                               |
//  |  Returns the end point radius of the off-mesh connection.				LH2'19|
//  +-----------------------------------------------------------------------------+
float NavMeshBuilder::GetOmcRadius( dtPolyRef ref )
{
	int omcIdx = GetOmcIndexFromPolyRef( ref, m_navMesh, m_offMeshUserIDs );
	if (omcIdx > -1) return m_offMeshRadii[omcIdx];
	else return 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::SetOmcRadius                                               |
//  |  Sets the end point radius of the off-mesh connection. Requires call to     |
//  |  ApplyChanges before being effective.									LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshBuilder::SetOmcRadius( dtPolyRef ref, float radius )
{
	int omcIdx = GetOmcIndexFromPolyRef( ref, m_navMesh, m_offMeshUserIDs );
	if (omcIdx > -1) m_offMeshRadii[omcIdx] = radius;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::GetOmcDirected                                             |
//  |  Returns the directedness of the off-mesh connection.					LH2'19|
//  +-----------------------------------------------------------------------------+
bool NavMeshBuilder::GetOmcDirected( dtPolyRef ref )
{
	int omcIdx = GetOmcIndexFromPolyRef( ref, m_navMesh, m_offMeshUserIDs );
	if (omcIdx > -1) return !(m_offMeshDirection[omcIdx] == DT_OFFMESH_CON_BIDIR);
	else return 0;
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::SetOmcRadius                                               |
//  |  Sets the directedness of the off-mesh connection. Requires call to	      |
//  |  ApplyChanges before being effective.									LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshBuilder::SetOmcDirected( dtPolyRef ref, bool unidirectional )
{
	int omcIdx = GetOmcIndexFromPolyRef( ref, m_navMesh, m_offMeshUserIDs );
	if (omcIdx > -1) m_offMeshDirection[omcIdx] = (unidirectional ? 0 : DT_OFFMESH_CON_BIDIR);
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::GetOmcVertex                                               |
//  |  Returns the vertex position of the off-mesh connection.				LH2'19|
//  +-----------------------------------------------------------------------------+
float3 NavMeshBuilder::GetOmcVertex( dtPolyRef ref, int vertexID )
{
	int omcIdx = GetOmcIndexFromPolyRef( ref, m_navMesh, m_offMeshUserIDs );
	if (omcIdx > -1) return m_offMeshVerts[omcIdx * 2 + vertexID];
	else return float3{ 0, 0, 0 };
}

//  +-----------------------------------------------------------------------------+
//  |  NavMeshBuilder::GetOmcVertex                                               |
//  |  Sets the vertex position of the off-mesh connection. Requires call to	  |
//  |  ApplyChanges before being effective.									LH2'19|
//  +-----------------------------------------------------------------------------+
void NavMeshBuilder::SetOmcVertex( dtPolyRef ref, int vertexID, float3 value )
{
	int omcIdx = GetOmcIndexFromPolyRef( ref, m_navMesh, m_offMeshUserIDs );
	if (omcIdx > -1) m_offMeshVerts[omcIdx * 2 + vertexID] = value;
}

} // namespace lighthouse2

// EOF