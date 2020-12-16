#include "bin.h"

/** (AL * NL + AR * NR) */
//void Bin::UpdateSurfaceAreaCost() {
//	this->cost = (this->boundsLeft.Area() * this->countLeft) + (this->boundsRight.Area() * this->countRight);
//}

void Bin::Clear() {
	this->count = 0;
	this->bounds = aabb();
}