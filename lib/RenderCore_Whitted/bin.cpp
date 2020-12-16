#include "bin.h"



/** (AL * NL + AR * NR) */
void Bin::UpdateSurfaceAreaCost() {
	this->cost = (this->boundsLeft.Area() * this->countLeft) + (this->boundsRight.Area() * this->countRight);
}

void Bin::Clear() {
	this->cost = 0;
	this->countLeft = 0;
	this->countRight = 0;
	this->splitPoint = 0;
	this->boundsLeft = aabb();
	this->boundsRight = aabb();
}