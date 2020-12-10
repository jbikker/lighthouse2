#include "bvh.h"
#include "bvhnode.h"
#include "vector"

BVH::BVH(int amountTriangles) {
	this->pool = new BVHNode[amountTriangles * 2 - 1];
	this->root = &this->pool[0];
	this->poolPtr = 2;

}