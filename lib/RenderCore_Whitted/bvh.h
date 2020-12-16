#pragma once

#include "core_settings.h"

class BVHNode;
class Bin;

class BVH
{
public:
	static int binCount;
	static Bin* bins;

	BVHNode* pool;
	BVHNode* root;
	int poolPtr;
	int* triangleIndices;
	BVH(int triangleIndex, int triangleCount);
};

