
#include <irrKlang.h>
#include <stdio.h>
#include <string.h>
#include "CIrrKlangAudioStreamLoaderMP3.h"

using namespace irrklang;

// this is the only function needed to be implemented for the plugin, it gets
// called by irrKlang when loaded.
// In this plugin, we create an audiostream loader class and register
// it at the engine, but a plugin can do anything.
// Be sure to name the function 'irrKlangPluginInit' and let the dll start with 'ikp'.

#ifdef WIN32
// Windows version
__declspec(dllexport) void __stdcall irrKlangPluginInit(ISoundEngine* engine, const char* version)
#else
// Linux version
void irrKlangPluginInit(ISoundEngine* engine, const char* version)
#endif
{
	// do some version security check to be sure that this plugin isn't begin used
	// by some newer irrKlang version with changed interfaces which could possibily
	// cause crashes.

	if (strcmp(version, IRR_KLANG_VERSION))
	{
		printf("This MP3 plugin only supports irrKlang version %s, mp3 playback disabled.\n", IRR_KLANG_VERSION);
		return;
	}

	// create and register the loader

	CIrrKlangAudioStreamLoaderMP3* loader = new CIrrKlangAudioStreamLoaderMP3();
	engine->registerAudioStreamLoader(loader);
	loader->drop();

	// that's it, that's all.
}

