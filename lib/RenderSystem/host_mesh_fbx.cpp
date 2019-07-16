/* host_mesh_fbx.cpp - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

#pragma comment(lib, "../fbxsdk/lib/libfbxsdk.lib" )

static FbxArray<FbxString*> animationStackNames; // TODO: not on global scope please.
static bool IS_SKELETON_ENABLED = false; // ??
static float globalScale = 1.0f, inverseGlobalScale = 1.0f / globalScale;
static char assetDirectory[1024];
static mat4 meshTransform;

std::vector<FBXImporter::FbxSceneAnimationData*> FBXImporter::fbxAnimData;

vector<HostMesh*> FBXImporter::Import( const char* fileName /* includes path */, const char* directory, const mat4& transform, int startingObjectIndex )
{
	// settings
	meshTransform = transform;

	// initialization
	FbxManager* sdkManager = FbxManager::Create();
	FbxIOSettings* iosettings = FbxIOSettings::Create( sdkManager, IOSROOT );
	sdkManager->SetIOSettings( iosettings );

	// importing
	vector<HostMesh*> meshes;
	FbxImporter* importer = FbxImporter::Create( sdkManager, "" );
	if (!importer->Initialize( fileName, -1, sdkManager->GetIOSettings() ))
	{
		printf( "FBX import failed: %s \n", importer->GetStatus().GetErrorString() ); return meshes;
	}
	strcpy_s( assetDirectory, fileName );
	assetDirectory[strlen( assetDirectory ) - 1] = 'm';
	strcat_s( assetDirectory, "\\" );
	FbxScene* scene = FbxScene::Create( sdkManager, fileName );
	importer->Import( scene );

#if 0
	// helper mesh for creating skeleton view
	FbxScene* skeletonScene = FbxScene::Create( sdkManager, "skeleton" );
	importer->Initialize( "data/limbNode/limbNode.fbx", -1, sdkManager->GetIOSettings() );
	importer->Import( skeletonScene );
	HostMesh* limbNodeMesh = CreateMesh( skeletonScene->GetNode( 1 ) );
#endif

	// polygon conversion
	FbxGeometryConverter converter( sdkManager );
	converter.Triangulate( scene, true );

#if 0
	converter.Triangulate( skeletonScene, true );
#endif

	// axis system conversion
	FbxAxisSystem sceneAxisSystem = scene->GetGlobalSettings().GetAxisSystem();
	FbxAxisSystem lighthouseAxisSystem = FbxAxisSystem( FbxAxisSystem::eYAxis, FbxAxisSystem::eParityOdd, FbxAxisSystem::eRightHanded );
	if (sceneAxisSystem != lighthouseAxisSystem)
	{
		lighthouseAxisSystem.ConvertScene( scene );
	}
	if (false) // (flipYZ)
	{
		FbxAxisSystem axisSystem = FbxAxisSystem( FbxAxisSystem::eZAxis, FbxAxisSystem::eParityOdd, FbxAxisSystem::eRightHanded ); //LightHouse axis system (+ mirrored X-axis)			
		axisSystem.ConvertScene( scene );
	}

	for (int i = 0; i < scene->GetNodeCount(); i++)
	{
		FbxNode* node = scene->GetNode( i );
		if (node->GetNodeAttribute() != NULL)
		{
			if (node->GetNodeAttribute()->GetAttributeType() == FbxNodeAttribute::eMesh)
				meshes.push_back( CreateMesh( node ) );
#if 0
			else if (node->GetNodeAttribute()->GetAttributeType() == FbxNodeAttribute::EType::eSkeleton && IS_SKELETON_ENABLED)
				meshes.push_back( limbNodeMesh );
#endif
			else if (node->GetNodeAttribute()->GetAttributeType() == FbxNodeAttribute::eLight)
			{
				// TODO: push back light
			}
		}
	}

	// get animation stack names
	FBXImporter::fbxAnimData.push_back( new FbxSceneAnimationData( fileName ) );
	FBXImporter::fbxAnimData.back()->positionOffset = make_float3( 0 );
	FBXImporter::fbxAnimData.back()->flipYZ = false; // flipYZ;
	scene->FillAnimStackNameArray( animationStackNames );
	if (FillAnimationStack( scene, startingObjectIndex )) for (int i = 0; i < meshes.size(); i++) meshes[i]->isAnimated = true;

	// cleanup and report back to caller
	importer->Destroy();
	iosettings->Destroy();
	sdkManager->Destroy();
	return meshes;
}

bool FBXImporter::FillAnimationStack( FbxScene* scene, int startingObjectIndex )
{
	bool hasAnimationStack = false;
	for (int i = 0; i < animationStackNames.Size(); i++)
	{
		FBXImporter::fbxAnimData.back()->animationStackList.push_back( new AnimationStack( animationStackNames[i]->Buffer() ) );
		FbxAnimStack* animationStack = scene->FindMember<FbxAnimStack>( animationStackNames[i]->Buffer() );
		FbxTakeInfo* takeInfo = scene->GetTakeInfo( animationStack->GetName() );
		if (takeInfo == NULL) continue;
		scene->SetCurrentAnimationStack( animationStack );
		FBXImporter::fbxAnimData.back()->animationStackList.back()->totalFrames = takeInfo->mLocalTimeSpan.GetStop().GetFrameCount() - takeInfo->mLocalTimeSpan.GetStart().GetFrameCount();
		for (int objectId = startingObjectIndex, j = 1; j < scene->GetNodeCount(); j++)
		{
			FbxNode* node = scene->GetNode( j );
			if (node->GetNodeAttribute() == NULL) continue;
			if (node->GetNodeAttribute()->GetAttributeType() == FbxNodeAttribute::eMesh)
				FBXImporter::fbxAnimData.back()->animationStackList.back()->nodeAnimData.push_back( FillNodeAnimationData( node, takeInfo, objectId++, false, HasDeformation( node ) ) );
#if 0
			else if (node->GetNodeAttribute()->GetAttributeType() == FbxNodeAttribute::EType::eSkeleton && IS_SKELETON_ENABLED)
				FBXImporter::fbxAnimData.back()->animationStackList.back()->nodeAnimData.push_back( FillNodeAnimationData( node, takeInfo, objectId++, true, false ) );
#endif
			else if (node->GetNodeAttribute()->GetAttributeType() == FbxNodeAttribute::EType::eCamera)
				FBXImporter::fbxAnimData.back()->animationStackList.back()->camAnimData.push_back( FillCameraAnimationData( node, takeInfo, objectId++ ) );
		}
		hasAnimationStack = true;
	}
	return hasAnimationStack;
}

FBXImporter::NodeAnimationData* FBXImporter::FillNodeAnimationData( FbxNode* node, FbxTakeInfo* takeInfo, int objectId, bool isSkeleton, bool hasDeformation )
{
	NodeAnimationData* nodeAnimationData = new NodeAnimationData( objectId, isSkeleton, hasDeformation );
	FbxAMatrix M = node->EvaluateGlobalTransform();
	nodeAnimationData->initialTranslation = make_float4( -(float)M.GetT()[0], (float)M.GetT()[1], (float)M.GetT()[2], 1 ) * inverseGlobalScale;
	nodeAnimationData->initialRotation = make_float4( -(float)M.GetR()[0], (float)M.GetR()[1], (float)M.GetR()[2], 1 );
	nodeAnimationData->initialScale = make_float4( -(float)M.GetS()[0], (float)M.GetS()[1], (float)M.GetS()[2], 1 ) * inverseGlobalScale;
	for (__int64 i = takeInfo->mLocalTimeSpan.GetStart().GetFrameCount(); i <= takeInfo->mLocalTimeSpan.GetStop().GetFrameCount(); i++)
	{
		FbxTime currentTime;
		currentTime.SetFrame( i );
		FbxAMatrix M = node->EvaluateGlobalTransform( currentTime );
		float4 translation, rotation, scale;
		translation = make_float4( -(float)M.GetT()[0], (float)M.GetT()[1], (float)M.GetT()[2], 1 ) * inverseGlobalScale;
		rotation = make_float4( -(float)M.GetR()[0], (float)M.GetR()[1], (float)M.GetR()[2], 1 );
		scale = make_float4( -(float)M.GetS()[0], (float)M.GetS()[1], (float)M.GetS()[2], 1 ) * inverseGlobalScale;
		nodeAnimationData->translationList.push_back( translation );
		nodeAnimationData->rotationList.push_back( rotation );
		nodeAnimationData->scaleList.push_back( scale );
		if (hasDeformation)
		{
			FbxMesh* mesh = node->GetMesh();
			uint vertexCount = mesh->GetControlPointsCount();
			FbxAMatrix vertexMatrix = node->EvaluateGlobalTransform() * GetGeometry( node );
			FbxVector4* vertexArray = new FbxVector4[vertexCount];
			memcpy( vertexArray, mesh->GetControlPoints(), vertexCount * sizeof( FbxVector4 ) );
			// check for vertex cache file
			if (node->GetMesh()->GetDeformerCount( FbxDeformer::eVertexCache ) && (static_cast<FbxVertexCacheDeformer*>(node->GetMesh()->GetDeformer( 0, FbxDeformer::eVertexCache )))->Active.Get())
			{
				FbxVertexCacheDeformer* deformer = static_cast<FbxVertexCacheDeformer*>(mesh->GetDeformer( 0, FbxDeformer::eVertexCache ));
				FbxCache* cache = deformer->GetCache();
				cache->OpenFileForRead();
				int channelIndex = cache->GetChannelIndex( deformer->Channel.Get() );
				float* vertexBuffer = NULL;
				uint bufferSize = 0;
				if (cache->Read( &vertexBuffer, bufferSize, currentTime, channelIndex )) for (uint j = 0; j < vertexCount; j++)
					vertexArray[j].mData[0] = vertexBuffer[j * 3 + 0] * inverseGlobalScale,
					vertexArray[j].mData[1] = vertexBuffer[j * 3 + 1] * inverseGlobalScale,
					vertexArray[j].mData[2] = vertexBuffer[j * 3 + 2] * inverseGlobalScale,
					vertexArray[j] = vertexMatrix.MultT( vertexArray[j] );
				nodeAnimationData->verticesCache.push_back( FillVertices( mesh, vertexArray ) );
			}
			else
			{
				const int skinCount = mesh->GetDeformerCount( FbxDeformer::eSkin );
				int clusterCount = 0;
				for (int j = 0; j < skinCount; j++) clusterCount += ((FbxSkin*)(mesh->GetDeformer( j, FbxDeformer::eSkin )))->GetClusterCount();
				if (clusterCount > 0)
				{
					FbxSkin* skinDeformer = (FbxSkin*)mesh->GetDeformer( 0, FbxDeformer::eSkin );
					FbxSkin::EType skinningType = skinDeformer->GetSkinningType();
					if (skinningType == FbxSkin::eRigid)
					{
						FbxVector4* vertexBuffer = ComputeLinearDeformation( mesh, vertexCount, skinCount, currentTime, vertexMatrix, vertexArray );
						nodeAnimationData->verticesCache.push_back( FillVertices( mesh, vertexBuffer ) );
					}
					else if (skinningType == FbxSkin::eDualQuaternion)
					{
						FbxVector4* vertexBuffer = ComputeDualQuaternionDeformation( mesh, vertexCount, skinCount, currentTime, vertexMatrix, vertexArray );
						nodeAnimationData->verticesCache.push_back( FillVertices( mesh, vertexBuffer ) );
					}
					else if (skinningType == FbxSkin::eBlend)
					{
						FbxVector4* vertexBufferLinear = ComputeLinearDeformation( mesh, vertexCount, skinCount, currentTime, vertexMatrix, vertexArray );
						FbxVector4* vertexBufferDualQuaterion = ComputeDualQuaternionDeformation( mesh, vertexCount, skinCount, currentTime, vertexMatrix, vertexArray );
						FbxVector4* vertexBuffer = new FbxVector4[vertexCount];
						int blendWeightsCount = skinDeformer->GetControlPointIndicesCount();
						for (int j = 0; j < blendWeightsCount; j++)
						{
							double blendWeight = skinDeformer->GetControlPointBlendWeights()[j];
							vertexBuffer[j] = vertexBufferDualQuaterion[j] * blendWeight + vertexBufferLinear[j] * (1 - blendWeight);
						}
						nodeAnimationData->verticesCache.push_back( FillVertices( mesh, vertexBuffer ) );
					}
				}
			}
		}
	}
	return nodeAnimationData;
}

FbxVector4* FBXImporter::ComputeLinearDeformation( FbxMesh* mesh, uint vertexCount, const int skinCount, FbxTime currentTime, FbxAMatrix vertexMatrix, FbxVector4* vertexArray )
{
	FbxCluster::ELinkMode clusterMode = ((FbxSkin*)mesh->GetDeformer( 0, FbxDeformer::eSkin ))->GetCluster( 0 )->GetLinkMode();
	FbxAMatrix* clusterDeformation = new FbxAMatrix[vertexCount];
	memset( clusterDeformation, 0, vertexCount * sizeof( FbxAMatrix ) );
	double* clusterWeight = new double[vertexCount];
	memset( clusterWeight, 0, vertexCount * sizeof( double ) );
	for (int i = 0; i < skinCount; i++)
	{
		FbxSkin* skinDeformer = (FbxSkin *)mesh->GetDeformer( i, FbxDeformer::eSkin );
		int clusterCount = skinDeformer->GetClusterCount();
		for (int j = 0; j < clusterCount; j++)
		{
			FbxCluster* cluster = skinDeformer->GetCluster( j );
			FbxAMatrix vertexTransformMatrix = ComputeClusterDeformation( mesh, cluster, currentTime );
			int vertexIndexCount = cluster->GetControlPointIndicesCount();
			for (int k = 0; k < vertexIndexCount; k++)
			{
				int index = cluster->GetControlPointIndices()[k];
				double weight = cluster->GetControlPointWeights()[k];
				FbxAMatrix influence = vertexTransformMatrix;;
				MatrixScale( influence, weight );
				MatrixAdd( clusterDeformation[index], influence );
				clusterWeight[index] += weight;
			}
		}
	}
	FbxVector4* vertexBuffer = new FbxVector4[vertexCount];
	for (uint i = 0; i < vertexCount; i++)
	{
		vertexBuffer[i] = vertexMatrix.MultT( clusterDeformation[i].MultT( vertexArray[i] ) );
		if (clusterWeight[i] != 0.0) vertexBuffer[i] /= (clusterWeight[i] * globalScale);
	}
	return vertexBuffer;
}

FbxVector4* FBXImporter::ComputeDualQuaternionDeformation( FbxMesh* mesh, uint vertexCount, const int skinCount, FbxTime currentTime, FbxAMatrix vertexMatrix, FbxVector4* vertexArray )
{
	FbxCluster::ELinkMode clusterMode = ((FbxSkin*)mesh->GetDeformer( 0, FbxDeformer::eSkin ))->GetCluster( 0 )->GetLinkMode();
	FbxDualQuaternion* clusterDeformation = new FbxDualQuaternion[vertexCount];
	memset( clusterDeformation, 0, vertexCount * sizeof( FbxDualQuaternion ) );
	double* clusterWeight = new double[vertexCount];
	memset( clusterWeight, 0, vertexCount * sizeof( double ) );
	for (int i = 0; i < skinCount; i++)
	{
		FbxSkin* skinDeformer = (FbxSkin *)mesh->GetDeformer( i, FbxDeformer::eSkin );
		int clusterCount = skinDeformer->GetClusterCount();
		for (int j = 0; j < clusterCount; j++)
		{
			FbxCluster* cluster = skinDeformer->GetCluster( j );
			FbxAMatrix vertexTransformMatrix = ComputeClusterDeformation( mesh, cluster, currentTime );
			FbxQuaternion quaternion = vertexTransformMatrix.GetQ();
			FbxVector4 translation = vertexTransformMatrix.GetT();
			FbxDualQuaternion dualQuaternion( quaternion, translation );
			int vertexIndexCount = cluster->GetControlPointIndicesCount();
			for (int k = 0; k < vertexIndexCount; k++)
			{
				int index = cluster->GetControlPointIndices()[k];
				double weight = cluster->GetControlPointWeights()[k];
				FbxDualQuaternion influence = dualQuaternion * weight;
				if (j == 0) clusterDeformation[index] = influence; else
				{
					double sign = clusterDeformation[index].GetFirstQuaternion().DotProduct( dualQuaternion.GetFirstQuaternion() );
					if (sign >= 0.0) clusterDeformation[index] += influence; else clusterDeformation[index] -= influence;
				}
				clusterWeight[index] += weight;
			}
		}
	}
	FbxVector4* vertexBuffer = new FbxVector4[vertexCount];
	for (uint i = 0; i < vertexCount; i++)
	{
		clusterDeformation[i].Normalize();
		vertexBuffer[i] = vertexMatrix.MultT( clusterDeformation[i].Deform( vertexArray[i] ) );
		if (clusterWeight[i] != 0.0) vertexBuffer[i] /= (clusterWeight[i] * globalScale);
	}
	return vertexBuffer;
}

vector<float3> FBXImporter::FillVertices( FbxMesh* mesh, FbxVector4* vertexBuffer )
{
	vector<float3> vertices;
	int indexCount = mesh->GetPolygonVertexCount();
	int* indices = mesh->GetPolygonVertices();
	for (int index = 0; index < indexCount; index++)
		vertices.push_back( make_float3( -(float)vertexBuffer[indices[index]].mData[0], (float)vertexBuffer[indices[index]].mData[1], (float)vertexBuffer[indices[index]].mData[2] ) );
	return vertices;
}

FBXImporter::CameraAnimationData* FBXImporter::FillCameraAnimationData( FbxNode* node, FbxTakeInfo* takeInfo, int objectId )
{
	CameraAnimationData* cameraAnimationData = new CameraAnimationData( objectId );
	FbxCamera* camera = (FbxCamera*)node->GetNodeAttribute();
	FbxAMatrix initialTransformOffset = node->EvaluateGlobalTransform();
	cameraAnimationData->initialTranslation = make_float4( -(float)initialTransformOffset.GetT()[0], (float)initialTransformOffset.GetT()[1], (float)initialTransformOffset.GetT()[2], 1 ) * inverseGlobalScale;
	cameraAnimationData->initialRotation = make_float4( -(float)initialTransformOffset.GetR()[0], (float)initialTransformOffset.GetR()[1], (float)initialTransformOffset.GetR()[2], 1 );
	for (FbxLongLong i = takeInfo->mLocalTimeSpan.GetStart().GetFrameCount(); i <= takeInfo->mLocalTimeSpan.GetStop().GetFrameCount(); i++)
	{
		FbxTime currentTime;
		currentTime.SetFrame( i );
		FbxVector4 cameraPosition = camera->EvaluatePosition( currentTime );
		FbxVector4 cameraLookAt = camera->EvaluateLookAtPosition( currentTime );
		const float4 position = make_float4( -(float)cameraPosition[0], (float)cameraPosition[1], (float)cameraPosition[2], 1 ) * inverseGlobalScale;
		const float4 lookAt = make_float4( -(float)cameraLookAt[0], (float)cameraLookAt[1], (float)cameraLookAt[2], 1 ) * inverseGlobalScale;
		cameraAnimationData->positionList.push_back( position );
		cameraAnimationData->lookAtList.push_back( lookAt );
	}
	return cameraAnimationData;
}

HostMesh* FBXImporter::CreateMesh( FbxNode* node )
{
	HostMesh* mesh = new HostMesh();
	mesh->name = node->GetName();
	HostMaterial* material = CreateMaterial( node );
	// mesh->material = material->ID;
	vector<float3> vlist, nlist;
	vector<float2> uvlist;
	vector<uint> indexList;

	// final matrices
	FbxAMatrix vertexMatrix = node->EvaluateGlobalTransform() * GetGeometry( node );
	FbxAMatrix normalMatrix = FbxAMatrix( FbxVector4( 0.0, 0.0, 0.0 ), vertexMatrix.GetR(), vertexMatrix.GetS() );

	for (int i = 0; i < node->GetNodeAttributeCount(); i++)
	{
		FbxMesh* fbxMesh = (FbxMesh*)node->GetNodeAttributeByIndex( i );
		FbxLayer* layer = fbxMesh->GetLayer( 0 );

		// indices
		int indexCount = fbxMesh->GetPolygonVertexCount();
		for (int j = 0; j < indexCount; j++) indexList.push_back( (uint)j );

		// vertices
		FbxVector4* controlPoints = fbxMesh->GetControlPoints();
		int* indices = fbxMesh->GetPolygonVertices();
		for (int j = 0; j < indexCount; j++)
		{
			FbxVector4 controlPoint = vertexMatrix.MultT( controlPoints[indices[j]] );
			const float3 vertexPos = make_float3( -(float)controlPoint.mData[0], (float)controlPoint.mData[1], (float)controlPoint.mData[2] );
			vlist.push_back( make_float3( make_float4( vertexPos, 1 ) * meshTransform ) );
		}

		// normals
		if (layer->GetNormals())
		{
			FbxVector4* normals = (FbxVector4*)layer->GetNormals()->GetDirectArray().GetLocked();
			for (int j = 0; j < indexCount; j++)
			{
				FbxVector4 normal = normalMatrix.MultT( normals[j] );
				nlist.push_back( make_float3( -(float)normal.mData[0], (float)normal.mData[1], (float)normal.mData[2] ) );
			}
			layer->GetNormals()->GetDirectArray().Release( (void**)&normals );
		}

		// uv's
		FbxStringList uvSetNameList;
		node->GetMesh()->GetUVSetNames( uvSetNameList );
		for (int j = 0; j < uvSetNameList.GetCount(); j++)
		{
			char* uvSetName = uvSetNameList.GetStringAt( j );
			FbxGeometryElementUV* uvElement = node->GetMesh()->GetElementUV( uvSetName );
			bool useIndex = uvElement->GetReferenceMode() != FbxGeometryElement::eDirect;
			int indexCount = (useIndex) ? uvElement->GetIndexArray().GetCount() : 0;
			int polyCount = node->GetMesh()->GetPolygonCount();
			if (uvElement->GetMappingMode() == FbxGeometryElement::eByControlPoint)
			{
				for (int k = 0; k < polyCount; k++)
				{
					int polySize = node->GetMesh()->GetPolygonSize( k );
					for (int l = 0; l < polySize; l++)
					{
						FbxVector2 uv;
						int polyVertexIndex = node->GetMesh()->GetPolygonVertex( k, l );
						int uvIndex = useIndex ? uvElement->GetIndexArray().GetAt( polyVertexIndex ) : polyVertexIndex;
						uv = uvElement->GetDirectArray().GetAt( uvIndex );
						uvlist.push_back( make_float2( (float)uv.mData[0], (float)uv.mData[1] ) );
					}
				}
			}
			else if (uvElement->GetMappingMode() == FbxGeometryElement::eByPolygonVertex)
			{
				int polyIndexCounter = 0;
				for (int k = 0; k < polyCount; k++)
				{
					int polySize = node->GetMesh()->GetPolygonSize( k );
					for (int l = 0; l < polySize; l++)
					{
						if (polyIndexCounter < indexCount)
						{
							FbxVector2 uv;
							int uvIndex = useIndex ? uvElement->GetIndexArray().GetAt( polyIndexCounter ) : polyIndexCounter;
							uv = uvElement->GetDirectArray().GetAt( uvIndex );
							uvlist.push_back( make_float2( (float)uv.mData[0], (float)uv.mData[1] ) );
							polyIndexCounter++;
						}
					}
				}
			}
		}
	}
	// create actual mesh
	mesh->triangles.resize( indexList.size() / 3 );
	for (size_t i = 0; i < indexList.size(); i += 3)
	{
		HostTri& tri = mesh->triangles[i / 3];
		const uint v0idx = indexList[i + 0];
		const uint v1idx = indexList[i + 1];
		const uint v2idx = indexList[i + 2];
		const int vertIdx = (int)mesh->vertices.size();
		mesh->indices.push_back( make_uint3( vertIdx, vertIdx + 1, vertIdx + 2 ) );
		const float3 v0pos = vlist[v0idx];
		const float3 v1pos = vlist[v1idx];
		const float3 v2pos = vlist[v2idx];
		mesh->vertices.push_back( make_float4( v0pos, 1 ) );
		mesh->vertices.push_back( make_float4( v1pos, 1 ) );
		mesh->vertices.push_back( make_float4( v2pos, 1 ) );
		const float3 N = normalize( cross( v1pos - v0pos, v2pos - v0pos ) );
		tri.Nx = N.x, tri.Ny = N.y, tri.Nz = N.z;
		tri.vertex0 = v0pos;
		tri.vertex1 = v1pos;
		tri.vertex2 = v2pos;
		tri.alpha = make_float3( 0 ); // make_float3( alphas[v0idx], alphas[v1idx], alphas[v2idx] );
		if (nlist.size() > 0)
			tri.vN0 = nlist[v0idx],
			tri.vN1 = nlist[v1idx],
			tri.vN2 = nlist[v2idx];
		if (uvlist.size() > 0)
			tri.u0 = uvlist[v0idx].x, tri.v0 = uvlist[v0idx].y,
			tri.u1 = uvlist[v1idx].x, tri.v1 = uvlist[v1idx].y,
			tri.u2 = uvlist[v2idx].x, tri.v2 = uvlist[v2idx].y;
		tri.material = material->ID;
	}
	return mesh;
}

HostMaterial* FBXImporter::CreateMaterial( FbxNode* node )
{
	HostMaterial* material;
	if (node->GetMaterialCount() > 0)
	{
		FbxSurfaceMaterial* fbxMaterial = node->GetMaterial( 0 );
		int matID = HostScene::FindMaterialID( fbxMaterial->GetName() );
		if (matID != -1)
		{
			material = HostScene::materials[matID];
			return material;
		}
		matID = HostScene::AddMaterial( make_float3( 0 ) );
		material = HostScene::materials[matID];
		material->name = fbxMaterial->GetName();
		FbxDouble3 property;
		property = ((FbxSurfaceLambert*)fbxMaterial)->Ambient;
		material->roughness = (property[0] + property[1] + property[2]) / 3;
		property = ((FbxSurfaceLambert*)fbxMaterial)->Diffuse;
		material->baseColor = make_float3( property[0], property[1], property[2] );
		property = ((FbxSurfaceLambert*)fbxMaterial)->Emissive;
		material->baseColor += make_float3( property[0], property[1], property[2] );
		if (HasDeformation( node )) material->flags = 0; else material->flags = HostMaterial::SMOOTH;
		// load texture from fbx diffuse property
		FbxProperty materialDiffuseProperty = fbxMaterial->FindProperty( fbxsdk::FbxSurfaceMaterial::sDiffuse );
		if (materialDiffuseProperty.IsValid())
		{
			if (materialDiffuseProperty.GetSrcObjectCount<FbxFileTexture>() > 0)
			{
				FbxFileTexture* fbxTexture = materialDiffuseProperty.GetSrcObject<FbxFileTexture>( 0 );
				FbxString textureFileName = FbxPathUtils::GetFileName( fbxTexture->GetFileName(), true );
				char textureFullPath[1024];
				strcpy_s( textureFullPath, assetDirectory );
				strcat_s( textureFullPath, textureFileName );
				material->map[TEXTURE0].textureID = HostScene::FindOrCreateTexture( textureFullPath );
			}
		}
	}
	else
	{
		material = HostScene::materials[HostScene::FindOrCreateMaterial( node->GetName() )];
	}
	return material;
}

void FBXImporter::MatrixScale( FbxAMatrix& matrix, double value )
{
	for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) matrix[i][j] *= value;
}

void FBXImporter::MatrixAdd( FbxAMatrix& destinationMatrix, FbxAMatrix& sourceMatrix )
{
	for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) destinationMatrix[i][j] += sourceMatrix[i][j];
}

bool FBXImporter::HasDeformation( FbxNode* node )
{
	bool hasVertexCache = node->GetMesh()->GetDeformerCount( FbxDeformer::eVertexCache ) && (static_cast<FbxVertexCacheDeformer*>(node->GetMesh()->GetDeformer( 0, FbxDeformer::eVertexCache )))->Active.Get();
	bool hasSkin = node->GetMesh()->GetDeformerCount( FbxDeformer::eSkin ) > 0;
	return hasVertexCache || hasSkin;
}

FbxAMatrix FBXImporter::ComputeClusterDeformation( FbxMesh* mesh, FbxCluster* cluster, FbxTime currentTime )
{
	FbxAMatrix referenceGlobalInitPosition;
	FbxAMatrix referenceGlobalCurrentPosition;
	FbxAMatrix associateGlobalInitPosition;
	FbxAMatrix associateGlobalCurrentPosition;
	FbxAMatrix clusterGlobalInitPosition;
	FbxAMatrix clusterGlobalCurrentPosition;
	FbxAMatrix referenceGeometry;
	FbxAMatrix associateGeometry;
	FbxAMatrix clusterGeometry;
	FbxAMatrix clusterRelativeInitPosition;
	FbxAMatrix clusterRelativeCurrentPositionInverse;
	cluster->GetTransformMatrix( referenceGlobalInitPosition );
	FbxAMatrix geometryOffset = GetGeometry( mesh->GetNode() );
	referenceGlobalCurrentPosition = mesh->GetNode()->EvaluateGlobalTransform() * geometryOffset;
	// multiply lReferenceGlobalInitPosition by Geometric Transformation
	referenceGeometry = GetGeometry( mesh->GetNode() );
	referenceGlobalInitPosition *= referenceGeometry;
	// get the link initial global position and the link current global position										
	cluster->GetTransformLinkMatrix( clusterGlobalInitPosition );
	clusterGlobalCurrentPosition = cluster->GetLink()->EvaluateGlobalTransform( currentTime );
	// compute the initial position of the link relative to the reference
	clusterRelativeInitPosition = clusterGlobalInitPosition.Inverse() * referenceGlobalInitPosition;
	// compute the current position of the link relative to the reference
	clusterRelativeCurrentPositionInverse = referenceGlobalCurrentPosition.Inverse() * clusterGlobalCurrentPosition;
	// compute the shift of the link relative to the reference
	return clusterRelativeCurrentPositionInverse * clusterRelativeInitPosition;
}

FbxAMatrix FBXImporter::GetGeometry( FbxNode* node )
{
	FbxVector4 translation = node->GetGeometricTranslation( FbxNode::eSourcePivot );
	FbxVector4 rotation = node->GetGeometricRotation( FbxNode::eSourcePivot );
	FbxVector4 scale = node->GetGeometricScaling( FbxNode::eSourcePivot );
	return FbxAMatrix( translation, rotation, scale );
}

// EOF