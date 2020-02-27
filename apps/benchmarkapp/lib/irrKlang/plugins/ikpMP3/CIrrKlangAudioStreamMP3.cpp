// Copyright (C) 2002-2007 Nikolaus Gebhardt
// Part of the code for this plugin for irrKlang is based on:
//  MP3 input for Audiere by Matt Campbell <mattcampbell@pobox.com>, based on
//  libavcodec from ffmpeg (http://ffmpeg.sourceforge.net/).
// See license.txt for license details of this plugin.

#include "CIrrKlangAudioStreamMP3.h"
#include <memory.h>
#include <stdlib.h> // free, malloc and realloc
#include <string.h>

namespace irrklang
{

CIrrKlangAudioStreamMP3::CIrrKlangAudioStreamMP3(IFileReader* file)
: File(file), TheMPAuDecContext(0), InputPosition(0), InputLength(0),
	DecodeBuffer(0), FirstFrameRead(false), EndOfFileReached(0),
	FileBegin(0), Position(0)
{
	if (File)
	{
		File->grab();

		TheMPAuDecContext = new MPAuDecContext();

		if (!TheMPAuDecContext || mpaudec_init(TheMPAuDecContext) < 0)
		{
			File->drop();
			File = 0;
			delete TheMPAuDecContext;
			TheMPAuDecContext = 0;
			return;
		}

		// init, get format

		DecodeBuffer = new ik_u8[MPAUDEC_MAX_AUDIO_FRAME_SIZE];

		if (File->getSize()>0)
		{
			// seekable file, now parse file to get size
			// (needed to make it possible for the engine to loop a stream correctly)

			skipID3IfNecessary();

			TheMPAuDecContext->parse_only = 1;
			Format.FrameCount = 0;

			while(!EndOfFileReached)
			{
				if (!decodeFrame())
					break;

				Format.FrameCount += TheMPAuDecContext->frame_size;

				if (!EndOfFileReached /*&& File->isSeekable()*/ )
				{
					// to be able to seek in the stream, store offsets and sizes

					SFramePositionData data;
					data.size = TheMPAuDecContext->frame_size;
					data.offset = File->getPos() - (InputLength - InputPosition) - TheMPAuDecContext->coded_frame_size;

					FramePositionData.push_back(data);
				}
			}

			TheMPAuDecContext->parse_only = 0;
			setPosition(0);
		}
		else
			decodeFrame(); // decode first frame to read audio format

		if (!TheMPAuDecContext->channels ||
			!TheMPAuDecContext->sample_rate )
		{
			File->drop();
			File = 0;
			delete TheMPAuDecContext;
			TheMPAuDecContext = 0;
			return;
		}
	}
}

CIrrKlangAudioStreamMP3::~CIrrKlangAudioStreamMP3()
{
	if (File)
		File->drop();

	if (TheMPAuDecContext)
	{
		mpaudec_clear(TheMPAuDecContext);
		delete TheMPAuDecContext;
	}

	delete [] DecodeBuffer;
}



//! returns format of the audio stream
SAudioStreamFormat CIrrKlangAudioStreamMP3::getFormat()
{
	return Format;
}


//! tells the audio stream to read n audio frames into the specified buffer
ik_s32 CIrrKlangAudioStreamMP3::readFrames(void* target, ik_s32 frameCountToRead)
{
	const int frameSize = Format.getFrameSize();

	int framesRead = 0;
	ik_u8* out = (ik_u8*)target;

	while (framesRead < frameCountToRead)
	{
		// no more samples?  ask the MP3 for more
		if (DecodedQueue.getSize() < frameSize)
		{
			if (!decodeFrame() || EndOfFileReached)
				return framesRead;

			// if the buffer is still empty, we are done
			if (DecodedQueue.getSize() < frameSize)
				return framesRead;
		}

		const int framesLeft = frameCountToRead - framesRead;
		const int dequeSize = DecodedQueue.getSize() / frameSize;
		const int framesToRead = framesLeft < dequeSize ? framesLeft : dequeSize;

		DecodedQueue.read(out, framesToRead * frameSize);

		out += framesToRead * frameSize;
		framesRead += framesToRead;
		Position += framesToRead;
	}

	return framesRead;
}



bool CIrrKlangAudioStreamMP3::decodeFrame()
{
    int outputSize = 0;

	while (!outputSize)
	{
		if (InputPosition == InputLength)
		{
			InputPosition = 0;
			InputLength = File->read(InputBuffer, IKP_MP3_INPUT_BUFFER_SIZE);

			if (InputLength == 0)
			{
				EndOfFileReached = true;
				return true;
			}
		}

		int rv = mpaudec_decode_frame( TheMPAuDecContext, (ik_s16*)DecodeBuffer,
									   &outputSize,
									   (ik_u8*)InputBuffer + InputPosition,
									   InputLength - InputPosition);

		if (rv < 0)
			return false;

		InputPosition += rv;
	} // end while

	if (!FirstFrameRead)
	{
		Format.ChannelCount = TheMPAuDecContext->channels;
		Format.SampleRate = TheMPAuDecContext->sample_rate;
		Format.SampleFormat = ESF_S16;
		Format.FrameCount = -1; // unknown lenght

		FirstFrameRead = true;
	}
	else
	if (TheMPAuDecContext->channels != Format.ChannelCount ||
		TheMPAuDecContext->sample_rate != Format.SampleRate)
	{
		// Can't handle format changes mid-stream.
		return false;
    }

	if (!TheMPAuDecContext->parse_only)
	{
		if (outputSize < 0)
		{
			// Couldn't decode this frame.  Too bad, already lost it.
			// This should only happen when seeking.

			outputSize = TheMPAuDecContext->frame_size;
			memset(DecodeBuffer, 0, outputSize * Format.getFrameSize());
		}

		DecodedQueue.write(DecodeBuffer, outputSize);
	}

    return true;
}



//! sets the position of the audio stream.
/** For example to let the stream be read from the beginning of the file again,
setPosition(0) would be called. This is usually done be the sound engine to
loop a stream after if has reached the end. Return true if sucessful and 0 if not. */
bool CIrrKlangAudioStreamMP3::setPosition(ik_s32 pos)
{
	if (!File || !TheMPAuDecContext)
		return false;

	if (pos == 0)
	{
		// usually done for looping, just reset to start

		File->seek(FileBegin); // skip possible ID3 header

		EndOfFileReached = false;

		DecodedQueue.clear();

		MPAuDecContext oldContext = *TheMPAuDecContext;

		mpaudec_clear(TheMPAuDecContext);
		mpaudec_init(TheMPAuDecContext);

		TheMPAuDecContext->bit_rate = oldContext.bit_rate;
		TheMPAuDecContext->channels = oldContext.channels;
		TheMPAuDecContext->frame_size = oldContext.frame_size;
		TheMPAuDecContext->sample_rate = oldContext.sample_rate;

		InputPosition = 0;
		InputLength = 0;
		Position = 0;
		CurrentFramePosition = 0;

		return true;
	}
	else
	{
		// user wants to seek in the stream, so do this here

		int scan_position = 0;
		int target_frame = 0;
		int frame_count = (int)FramePositionData.size();

		while (target_frame < frame_count)
		{
			int frame_size = FramePositionData[target_frame].size;

			if (pos <= scan_position + frame_size)
				break;
			else
			{
				scan_position += frame_size;
				target_frame++;
			}
		}


		const int MAX_FRAME_DEPENDENCY = 10;
		target_frame = std::max(0, target_frame - MAX_FRAME_DEPENDENCY);
		setPosition(0);

		File->seek(FramePositionData[target_frame].offset, false);

		int i;
		for (i = 0; i < target_frame; i++)
		{
			if (i>=(int)FramePositionData.size())
			{
				// internal error
				setPosition(0);
				return false;
			}

			Position += FramePositionData[i].size;
		}

		if (!decodeFrame() || EndOfFileReached)
		{
			setPosition(0);
			return false;
		}

		int frames_to_consume = pos - Position; // PCM frames now
		if (frames_to_consume > 0)
		{
			ik_u8 *buf = new ik_u8[frames_to_consume * Format.getFrameSize()];
			readFrames(buf, frames_to_consume);
			delete[] buf;
		}

      	return true;
	}

	return false;
}


CIrrKlangAudioStreamMP3::QueueBuffer::QueueBuffer()
{
	Capacity = 256;
	Size = 0;

	Buffer = (ik_u8*)malloc(Capacity);
}


CIrrKlangAudioStreamMP3::QueueBuffer::~QueueBuffer()
{
	free(Buffer);
}

int CIrrKlangAudioStreamMP3::QueueBuffer::getSize()
{
	return Size;
}

void CIrrKlangAudioStreamMP3::QueueBuffer::write(const void* buffer, int size)
{
	bool needRealloc = false;

	while (size + Size > Capacity)
	{
		Capacity *= 2;
		needRealloc = true;
	}

    if (needRealloc)
	{
        Buffer = (ik_u8*)realloc(Buffer, Capacity);
    }

	memcpy(Buffer + Size, buffer, size);
	Size += size;
}


int CIrrKlangAudioStreamMP3::QueueBuffer::read(void* buffer, int size)
{
	int toRead = size < Size ? size : Size;

	memcpy(buffer, Buffer, toRead);
	memmove(Buffer, Buffer + toRead, Size - toRead);

	Size -= toRead;
	return toRead;
}


void CIrrKlangAudioStreamMP3::QueueBuffer::clear()
{
	Size = 0;
}


void CIrrKlangAudioStreamMP3::skipID3IfNecessary()
{
	char header[10];
	int read = File->read(&header, 10);

	if (read == 10 &&
		header[0] == 'I' && header[1] == 'D' && header[2] == '3')
	{
		int versionMajor = header[3];
		int versionMinor = header[4];
		int flags = header[5];

		// IDv2 size looks like the following: ID3v2 size  4 * %0xxxxxxx.
		// Sick, but that's how it works.

		int size = 0;
		size  = (header[6] & 0x7f) << (3*7);
		size |= (header[7] & 0x7f) << (2*7);
		size |= (header[8] & 0x7f) << (1*7);
		size |= (header[9] & 0x7f) ;

		size += 10; // header size

		FileBegin = size;
		File->seek(FileBegin);
	}
	else
		File->seek(0);
}


} // end namespace irrklang
