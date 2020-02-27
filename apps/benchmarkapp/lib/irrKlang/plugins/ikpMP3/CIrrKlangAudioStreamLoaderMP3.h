// Copyright (C) 2002-2007 Nikolaus Gebhardt
// This file is part of the "irrKlang" library.
// For conditions of distribution and use, see copyright notice in irrKlang.h

#ifndef __C_IRRKLANG_AUDIO_STREAM_LOADER_MP3_H_INCLUDED__
#define __C_IRRKLANG_AUDIO_STREAM_LOADER_MP3_H_INCLUDED__

#include <ik_IAudioStreamLoader.h>

namespace irrklang
{
	//!	Class which is able to create an audio file stream from a file.
	class CIrrKlangAudioStreamLoaderMP3 : public IAudioStreamLoader
	{
	public:

		CIrrKlangAudioStreamLoaderMP3();

		//! Returns true if the file maybe is able to be loaded by this class.
		/** This decision should be based only on the file extension (e.g. ".wav") */
		virtual bool isALoadableFileExtension(const ik_c8* fileName);

		//! Creates an audio file input stream from a file
		/** \return Pointer to the created audio stream. Returns 0 if loading failed.
		If you no longer need the stream, you should call IAudioFileStream::drop().
		See IRefCounted::drop() for more information. */
		virtual IAudioStream* createAudioStream(irrklang::IFileReader* file);
	};

} // end namespace irrklang

#endif
