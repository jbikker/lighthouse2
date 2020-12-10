#include "bvhnode.h"
#include "whitted_ray_tracer.h"
#include "triangle.h"

void BVHNode::SubdivideNode(BVHNode* pool, int &poolPtr) {
	if (triangleIndices.size() < 3) return;
	this->left = &pool[poolPtr++];
	this->right = &pool[poolPtr++];
	this->PartitionTriangles();
	this->left->SubdivideNode(pool, poolPtr);
	this->right->SubdivideNode(pool, poolPtr);
	this->isLeaf = false;
}

void BVHNode::PartitionTriangles() {
	int axis = this->bounds.LongestAxis();
	float splitPoint = this->bounds.Center(axis);

	for (int i = 0; i < this->triangleIndices.size(); i++) {
		int triangleIndex = this->triangleIndices[i];
		Triangle* triangle = WhittedRayTracer::scene[triangleIndex];

		float trianglePoint;
		if (axis == 0) {
			trianglePoint = triangle->centroid.x;
		} else if (axis == 1) {
			trianglePoint = triangle->centroid.y;
		} else {
			trianglePoint = triangle->centroid.z;
		}

		if (trianglePoint < splitPoint) {
			left->triangleIndices.push_back(triangleIndex);
		} else {
			right->triangleIndices.push_back(triangleIndex);
		}
	}

	left->UpdateBounds();
	right->UpdateBounds();
}

void BVHNode::UpdateBounds() {
	this->bounds = aabb();
	
	for (int i = 0; i < this->triangleIndices.size(); i++) {
		Triangle* triangle = WhittedRayTracer::scene[this->triangleIndices[i]];

		if (triangle->v0.x < this->bounds.bmin[0]) {
			this->bounds.bmin[0] = triangle->v0.x;
		}
		if (triangle->v0.x > this->bounds.bmax[0]) {
			this->bounds.bmax[0] = triangle->v0.x;
		}
		if (triangle->v0.y < this->bounds.bmin[0]) {
			this->bounds.bmin[0] = triangle->v0.y;
		}
		if (triangle->v0.y > this->bounds.bmax[0]) {
			this->bounds.bmax[0] = triangle->v0.y;
		}
		if (triangle->v0.z < this->bounds.bmin[0]) {
			this->bounds.bmin[0] = triangle->v0.z;
		}
		if (triangle->v0.z > this->bounds.bmax[0]) {
			this->bounds.bmax[0] = triangle->v0.z;
		}
	}
}

bool BVHNode::Traverse(Ray &ray, float4 &color, BVHNode* root, uint &recursionDepth) {
	if (recursionDepth > WhittedRayTracer::recursionThreshold) { return false; }
	if (ray.IntersectBounds(this->bounds) == NULL) { return false; }
	if (this->isLeaf) { return this->IntersectTriangles(ray, color); }

	float leftNodeDist = ray.IntersectBounds(this->left->bounds);
	float rightNodeDist = ray.IntersectBounds(this->right->bounds);

	if (leftNodeDist == NULL && rightNodeDist == NULL) { return false; }

	if (leftNodeDist != NULL && rightNodeDist == NULL) {
		return this->left->Traverse(ray, color, recursionDepth);
	}

	if (leftNodeDist == NULL && rightNodeDist != NULL) {
		return this->right->Traverse(ray, color, recursionDepth);
	}

	float4 rayOrigin = ray.origin;
	float4 rayDirection = ray.direction;

	if (leftNodeDist < rightNodeDist) {
		bool hitLeft = this->left->Traverse(ray, color, recursionDepth);
		if (hitLeft) { return true; }

		ray.origin = rayOrigin;
		ray.direction = rayDirection; // Could be deleted when creating new ray
		return this->right->Traverse(ray, color, recursionDepth);
	} else {
		bool hitRight = this->right->Traverse(ray, color, recursionDepth);
		if (hitRight) { return true; }

		ray.origin = rayOrigin;
		ray.direction = rayDirection; // Could be deleted when creating new ray
		return this->left->Traverse(ray, color, recursionDepth);
	}
}

bool BVHNode::IntersectTriangles(Ray &ray, float4 &color, uint& recursionDepth) {
	tuple<Triangle*, float> nearestIntersection = ray.GetNearestIntersection(this->triangleIndices);
	Triangle* nearestTriangle = get<0>(nearestIntersection);
	float intersectionDistance = get<1>(nearestIntersection);

	if (intersectionDistance > 0) {
		float4 intersectionPoint = ray.GetIntersectionPoint(intersectionDistance);

		CoreMaterial* material = &WhittedRayTracer::materials[nearestTriangle->materialIndex];

		color += ray.DetermineColor(nearestTriangle, material, intersectionPoint, recursionDepth);
		return true;
	}

	return false;
}
