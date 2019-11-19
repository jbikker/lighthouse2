/* random.glsl - Copyright 2019 Utrecht University

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

#ifndef RANDOM_H
#define RANDOM_H

uint WangHash( uint s )
{
	s = (s ^ 61) ^ (s >> 16);
	s *= 9;
	s = s ^ (s >> 4);
	s *= 0x27d4eb2d, s = s ^ (s >> 15);
	return s;
}

uint RandomInt( inout uint s )
{
	s ^= s << 13;
	s ^= s >> 17;
	s ^= s << 5;
	return s;
}

float RandomFloat( inout uint s ) { return RandomInt( s ) * 2.3283064365387e-10f; }

#endif