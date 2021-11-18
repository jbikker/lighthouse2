/* core_api_base.cpp - Copyright 2019/2021 Utrecht University

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

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#ifdef APIENTRY
#undef APIENTRY
#endif

#include <windows.h>
static HMODULE module;

bool GetLastErrorString( char* buffer, DWORD length )
{
	auto err = GetLastError();

	if (!err)
		return false;

	// Assuming buffer is large enough for any error message.
	// Otherwise, use FORMAT_MESSAGE_ALLOCATE_BUFFER and LocalFree!
	auto ret = FormatMessage( FORMAT_MESSAGE_FROM_SYSTEM,
		NULL,
		err,
		MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ), // default language
		buffer,
		length,
		NULL );

	return ret > 0;
}

HMODULE LoadModule( const char* dllName )
{
	// try the development folder structure first
	string dllpath = "../../coredlls/";
#ifdef _DEBUG
	dllpath += "debug/";
#else
	dllpath += "release/";
#endif
	dllpath += dllName;
	module = LoadLibrary( dllpath.c_str() );
	if (!module)
	{
		char errstr[1024] = "FormatMessage failed (unknown error code?)";
		// Print informative info:
		if (GetLastErrorString( errstr, sizeof( errstr ) ))
			printf( "Failed to load %s: %s\tTrying local DLL\n", dllpath.c_str(), errstr );

		// see if the dll is perhaps in the current folder
		module = LoadLibrary( dllName );
		if (!module)
		{
			GetLastErrorString( errstr, sizeof( errstr ) );
			FATALERROR( "Unable to open %s: %s", dllName, errstr );
		}
	}
	return module;
}

FARPROC GetSymbol( HMODULE module, const char* symbol )
{
	return GetProcAddress( module, symbol );
}

#else

#include <dlfcn.h>
static void* module;

void* LoadModule( const char* dllName )
{
	auto dllpath = string( "lib" ) + string( dllName ) + string( ".so" );
	module = dlopen( dllpath.c_str(), RTLD_NOW | RTLD_GLOBAL );
	if (!module)
	{
		printf( "dlopen %s failed with %s\n", dllpath.c_str(), dlerror() );

		// see if the dll is perhaps in the current folder
		module = dlopen( dllName, RTLD_NOW | RTLD_GLOBAL );
		FATALERROR_IF( !module, "Failed to dlopen %s: %s", dllName, dlerror() );
	}
	return module;
}

void* GetSymbol( void* module, const char* symbol )
{
	return dlsym( module, symbol );
}

#endif

CoreAPI_Base* CoreAPI_Base::CreateCoreAPI( const char* coreName )
{
	if (!api)
	{
		module = LoadModule( coreName );
		createCore = (createCoreFunction)GetSymbol( module, "CreateCore" );
		FATALERROR_IF( !createCore, "Could not find CreateCore in library" );
		api = createCore();
		api->Init();
	}
	return api;
}

// EOF