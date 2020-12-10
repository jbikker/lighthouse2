#pragma once

#include "core_settings.h"
#include "bvhnode.h"
#include "triangle.h"

class BVH
{
public:
	BVHNode* pool;
	BVHNode* root;
	int poolPtr;
	BVH(int amountTriangles);
	
};

