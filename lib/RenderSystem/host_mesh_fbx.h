/* host_mesh_fbx.h - Copyright 2019 Utrecht University

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

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  FBXImporter                                                                |
//  |  By The Animation Guys.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
class FBXImporter
{
public:
	class AnimationData
	{
	public:
		virtual int GetObjectId() = 0;
		float4 initialTranslation, initialRotation;
		vector<float4> translationList, rotationList;
	private:
		int objectId;
	};
	class NodeAnimationData : public AnimationData
	{
	public:
		NodeAnimationData( int objcetId, bool isSkeleton = false, bool hasDeformation = false )
		{
			objectId = objectId;
			isSkeleton = isSkeleton;
			hasDeformation = hasDeformation;
		}
		int GetObjectId() { return this->objectId; }
		bool isSkeleton, hasDeformation;
		float4 initialTranslation, initialRotation, initialScale;
		vector<float4> translationList, rotationList, scaleList;
		vector<vector<float3>> verticesCache;
	private:
		int objectId;
	};
	class CameraAnimationData : public AnimationData
	{
	public:
		CameraAnimationData( int objectId )
		{
			objectId = objectId;
		}
		int GetObjectId() { return this->objectId; }
		float4 initialTranslation, initialRotation;
		vector<float4> positionList, lookAtList;
	private:
		int objectId;
	};
	class AnimationStack
	{
	public:
		AnimationStack( const char* name )
		{
			totalFrames = 0;
			framePointer = 0;
			currentCamera = 0;
			isCameraPlayback = false;
			strcpy_s( animationStackName, 1020, name );
		}
		char animationStackName[1024];
		__int64 totalFrames, framePointer;
		int currentCamera;
		bool isCameraPlayback;
		vector<NodeAnimationData*> nodeAnimData;
		vector<CameraAnimationData*> camAnimData;
	};
	class FbxSceneAnimationData
	{
	public:
		FbxSceneAnimationData( const char* name )
		{
			currentAnimationStack = 0;
			flipYZ = false;
			positionOffset = make_float3( 0 );
			strcpy_s( fbxSceneName, 1020, name );
		}
		char fbxSceneName[1024];
		bool flipYZ;
		int currentAnimationStack;
		float3 positionOffset;
		vector<AnimationStack*> animationStackList;
	};
	// methods
	static vector<HostMesh*> Import( const char* filePath, const char* directory, const mat4& transform = mat4(), int startingObjectIndex = 0 );
private:
	// methods
	static void MatrixScale( FbxAMatrix& matrix, double value );
	static void MatrixAdd( FbxAMatrix& destinationMatrix, FbxAMatrix & sourceMatrix );
	static FbxAMatrix ComputeClusterDeformation( FbxMesh* mesh, FbxCluster* cluster, FbxTime currentTime );
	static FbxAMatrix GetGeometry( FbxNode* node );
	static bool HasDeformation( FbxNode* node );
	static bool FillAnimationStack( FbxScene* scene, int startingObjectIndex );
	static NodeAnimationData* FillNodeAnimationData( FbxNode* node, FbxTakeInfo* takeInfo, int objectId, bool isSkeleton = false, bool hasDeformation = false );
	static FbxVector4* ComputeLinearDeformation( FbxMesh* mesh, uint vertexCount, const int skinCount, FbxTime currentTime, FbxAMatrix vertexMatrix, FbxVector4* vertexArray );
	static FbxVector4* ComputeDualQuaternionDeformation( FbxMesh* mesh, uint vertexCount, const int skinCount, FbxTime currentTime, FbxAMatrix vertexMatrix, FbxVector4* vertexArray );
	static vector<float3> FillVertices( FbxMesh* mesh, FbxVector4* vertexBuffer );
	static CameraAnimationData* FillCameraAnimationData( FbxNode* node, FbxTakeInfo* takeInfo, int objectId );
	static HostMesh* CreateMesh( FbxNode* node );
	static HostMaterial* CreateMaterial( FbxNode* node );
	// data members
	static vector<FbxSceneAnimationData*> fbxAnimData;
};

} // namespace lighthouse2

// EOF