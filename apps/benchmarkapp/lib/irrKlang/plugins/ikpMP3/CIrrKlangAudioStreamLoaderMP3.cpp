// Copyright (C) 2002-2007 Nikolaus Gebhardt
// This file is part of the "irrKlang" library.
// For conditions of distribution and use, see copyright notice in irrKlang.h

#include "CIrrKlangAudioStreamLoaderMP3.h"
#include "CIrrKlangAudioStreamMP3.h"
#include <string.h>


namespace irrklang
{


CIrrKlangAudioStreamLoaderMP3::CIrrKlangAudioStreamLoaderMP3()
{
}


//! Returns true if the file maybe is able to be loaded by this class.
bool CIrrKlangAudioStreamLoaderMP3::isALoadableFileExtension(const ik_c8* fileName)
{
	return strstr(fileName, ".mp3") != 0;
}


//! Creates an audio file input stream from a file
IAudioStream* CIrrKlangAudioStreamLoaderMP3::createAudioStream(irrklang::IFileReader* file)
{
	CIrrKlangAudioStreamMP3* stream = new CIrrKlangAudioStreamMP3(file);

	if (stream && !stream->isOK())
	{
		stream->drop();
		stream = 0;
	}

	return stream;
}


} // end namespace irrklang

