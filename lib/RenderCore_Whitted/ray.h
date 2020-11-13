#pragma once
class Ray
{
public:
	Ray(float4 orig, float4 dir);
	float4 origin;
	float4 direction;
};
