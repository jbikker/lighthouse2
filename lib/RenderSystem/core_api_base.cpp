/* core_api_base.cpp - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file contains the implementation of the base class for the core
   API.
*/

#include "rendersystem.h"

static CoreAPI_Base* api = 0;

typedef CoreAPI_Base* (*createCoreFunction)();
typedef void( *destroyCoreFunction )();
static createCoreFunction createCore;
static destroyCoreFunction destroyCore;
static HMODULE module;

CoreAPI_Base* CoreAPI_Base::CreateCoreAPI( const char* dllName )
{
	if (!api)
	{
		// try the development folder structure first
		string dllpath = "../../coredlls/";
	#ifdef _DEBUG
		dllpath += "debug/";
	#else
		dllpath += "release/";
	#endif
		dllpath += string( dllName );
		module = LoadLibrary( dllpath.c_str() );
		if (module == 0) 
		{
			// see if the dll is perhaps in the current folder
			module = LoadLibrary( dllName );
			if (module == 0) FatalError( "dll not found: %s", dllName );
		}
		createCore = (createCoreFunction)GetProcAddress( module, "CreateCore" );
		if (createCore == 0) FatalError( __FILE__, __LINE__, "could not find entrypoint in dll" );
		api = createCore();
		api->Init();
	}
	return api;
}

// EOF