/* bindings.h - Copyright 2019 Utrecht University

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

#ifndef BINDINGS_H
#define BINDINGS_H

// RT bindings
#define rtACCELERATION_STRUCTURE 0
#define rtCAMERA 1
#define rtPATH_STATES 2
#define rtPATH_ORIGINS 3
#define rtPATH_DIRECTIONS 4
#define rtPOTENTIAL_CONTRIBUTIONS 5
#define rtACCUMULATION_BUFFER 6
#define rtBLUENOISE 7

// Shade bindings
#define cCOUNTERS 0
#define cCAMERA 1
#define cPATH_STATES 2
#define cPATH_ORIGINS 3
#define cPATH_DIRECTIONS 4
#define cPATH_THROUGHPUTS 5
#define cPOTENTIAL_CONTRIBUTIONS 6
#define cSKYBOX 7
#define cMATERIALS 8
#define cTRIANGLES 9
#define cTRIANGLE_BUFFER_INDICES 10
#define cINVERSE_TRANSFORMS 11
#define cTEXTURE_ARGB32 12
#define cTEXTURE_ARGB128 13
#define cTEXTURE_NRM32 14
#define cACCUMULATION_BUFFER 15
#define cACCUMULATION_BUFFER 15
#define cAREALIGHT_BUFFER 16
#define	cPOINTLIGHT_BUFFER 17
#define cSPOTLIGHT_BUFFER 18
#define cDIRECTIONALLIGHT_BUFFER 19
#define cBLUENOISE 20

// Finalize bindings
#define fACCUMULATION_BUFFER 0
#define fUNIFORM_CONSTANTS 1
#define fOUTPUT 2

// Stage indices
#define STAGE_PRIMARY_RAY 0
#define STAGE_SECONDARY_RAY 1
#define STAGE_SHADOW_RAY 2
#define STAGE_SHADE 3
#define STAGE_FINALIZE 4

#define MAXPATHLENGTH 3
#define MAX_TRIANGLE_BUFFERS 65536
#define BLUENOISE
#endif