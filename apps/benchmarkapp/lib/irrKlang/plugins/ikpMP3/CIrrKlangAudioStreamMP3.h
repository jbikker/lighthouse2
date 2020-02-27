// Copyright (C) 2002-2007 Nikolaus Gebhardt
// Part of the code for this plugin for irrKlang is based on:
//  MP3 input for Audiere by Matt Campbell <mattcampbell@pobox.com>, based on
//  libavcodec from ffmpeg (http://ffmpeg.sourceforge.net/).
// See license.txt for license details of this plugin.

#ifndef __C_IRRKLANG_AUDIO_STREAM_MP3_H_INCLUDED__
#define __C_IRRKLANG_AUDIO_STREAM_MP3_H_INCLUDED__

#include <ik_IAudioStream.h>
#include <ik_IFileReader.h>
#include <vector>
#include "decoder/mpaudec.h"

namespace irrklang
{
	const int IKP_MP3_INPUT_BUFFER_SIZE = 4096;

	//!	Reads and decodes audio data into an usable audio stream for the ISoundEngine
	/** To extend irrKlang with new audio format decoders, the only thing needed to do
	is implementing the IAudioStream interface. All the code available in this class is only for
	mp3 decoding and may make this class look a bit more complicated then it actually is. */
	class CIrrKlangAudioStreamMP3 : public IAudioStream
	{
	public:

		CIrrKlangAudioStreamMP3(IFileReader* file);
		~CIrrKlangAudioStreamMP3();

		//! returns format of the audio stream
		virtual SAudioStreamFormat getFormat();

		//! tells the audio stream to read n audio frames into the specified buffer
		/** \param target: Target data buffer to the method will write the read frames into. The
		specified buffer will be getFormat().getFrameSize()*frameCount big.
		\param frameCount: amount of frames to be read.
		\returns Returns amount of frames really read. Should be frameCountToRead in most cases. */
		virtual ik_s32 readFrames(void* target, ik_s32 frameCountToRead);

		//! sets the position of the audio stream.
		/** For example to let the stream be read from the beginning of the file again,
		setPosition(0) would be called. This is usually done be the sound engine to
		loop a stream after if has reached the end. Return true if sucessful and 0 if not. */
		virtual bool setPosition(ik_s32 pos);

		// just for the CIrrKlangAudioStreamLoaderMP3 to let him know if loading worked
		bool isOK() { return File != 0; }

	protected:

		ik_s32 readFrameForMP3(void* target, ik_s32 frameCountToRead, bool parseOnly=false);
		bool decodeFrame();
		void skipID3IfNecessary();

		irrklang::IFileReader* File;
		SAudioStreamFormat Format;

		// mpaudec specific
		MPAuDecContext* TheMPAuDecContext;

		ik_u8 InputBuffer[IKP_MP3_INPUT_BUFFER_SIZE];

		int InputPosition;
		int InputLength;
		int Position;
		ik_u8* DecodeBuffer;
		ik_s32 FileBegin;
		ik_u32 CurrentFramePosition;

		bool FirstFrameRead;
		bool EndOfFileReached;

		// helper class for managing the streaming decoded audio data
		class QueueBuffer
		{
		public:	

			QueueBuffer();
			~QueueBuffer();

			int getSize();
			void write(const void* buffer, int size);
			int read(void* buffer, int size);
			void clear();

		private:

			ik_u8* Buffer;
			int Capacity;
			int Size;
		};

		struct SFramePositionData
		{
			int offset;
			int size;
		};

		std::vector<SFramePositionData> FramePositionData;
		QueueBuffer DecodedQueue;
	};


} // end namespace irrklang

#endif
