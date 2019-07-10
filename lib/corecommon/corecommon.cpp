#include "system.h"
#include <map>
#undef APIENTRY
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

using namespace std;

#include "corecommon.h"

// example of an exported variable
CORECOMMON_API int nCommonDLL = 0;

// example of an exported function
CORECOMMON_API int fnCommonDLL( void )
{
	return 42;
}

// constructor of a class that has been exported
CoreAPI_DLL* CoreAPI_DLL::CreateCoreAPI( const CoreID id )
{
	return 0;
}

// define the prototype for a function that should exist in the DLL that is used to create and return the plugin type in the DLL
typedef CoreAPI_DLL* (*fnCreateCore)(void);

// destroys the plugin type from the DLL before the library is unloaded
typedef void( *fnDestroyCore )(void);

#if 1

// copy of FatalError from system.cpp; TODO: don't use a copy
static void setfv( string& s, const char* fmt, va_list args )
{
	static char* buffer = 0;
	if (!buffer) buffer = new char[16384];
	int len = _vscprintf( fmt, args );
	if (!len) return;
	vsprintf_s( buffer, len + 1, fmt, args );
	s = buffer;
}

static void FatalError( const char* fmt, ... )
{
	string tmp;
	va_list args;
	va_start( args, fmt );
	setfv( tmp, fmt, args );
	va_end( args );
	printf( "\n%s\n", tmp.c_str() );
	MessageBox( NULL, tmp.c_str(), "Fatal error", MB_OK );
	while (1) exit( 0 );
}

#endif

CoreAPI_DLL* CoreManager::LoadCore( const string& coreName )
{
	CoreAPI_DLL* core = NULL;
	CoreMap::iterator iter = cores.find( coreName );
	if (iter != cores.end()) return iter->second; // already loaded
	// try to load the core
	HMODULE hModule = LoadLibrary( coreName.c_str() );
	if (hModule == NULL) FatalError( "Could not load core %s", coreName );
	fnCreateCore CreateCore = (fnCreateCore)GetProcAddress( hModule, "CreateCore" );
	if (CreateCore == NULL) FatalError( "Could not find symbol \"CreateCore\" in %s", coreName );
	// invoke the function to get the core from the DLL
	core = CreateCore();
	if (core == NULL) FatalError( "Could not load core from %s", coreName );
	core->SetName( coreName );
	// add the core and library to the maps
	cores.insert( CoreMap::value_type( coreName, core ) );
	libs.insert( LibraryMap::value_type( coreName, hModule ) );
	return core;
}

void CoreManager::UnloadCore( CoreAPI_DLL*& core )
{
	if (core == NULL) return;
	LibraryMap::iterator iter = libs.find( core->GetName() );
	if (iter == libs.end()) return;
	// remove the plugin from our plugin map
	cores.erase( core->GetName() );
	HMODULE hModule = iter->second;
	fnDestroyCore DestroyCore = (fnDestroyCore)GetProcAddress( hModule, "DestroyCore" );
	if (DestroyCore != NULL) DestroyCore(); else FatalError( "Unable to find symbol \"DestroyPlugin\" in library %s", core->GetName() );
	// unload the library and remove the library from the map
	FreeLibrary( hModule );
	libs.erase( iter );
	core = NULL;
}

// EOF