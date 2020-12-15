#pragma once

#include "core_settings.h"

class BVHNode;

class BVH
{
public:
	BVHNode* pool;
	BVHNode* root;
	int poolPtr;
	int* triangleIndices;
	BVH(int triangleIndex, int triangleCount);
};

