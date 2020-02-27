==========================================================================
The irrKlang SDK version 1.6
==========================================================================

  Welcome the irrKlang SDK.

  Content of this file:

  1. Directory structure overview
  2. How to start
  3. Release Notes
  4. .NET version dependencies
  5. License
  6. Contact



==========================================================================
1. Directory structure overview
==========================================================================

  You will find some directories after decompressing the archive in which
  came the SDK. These are:
  
  \bin         The compiled library irrKlang.DLL and some compiled demo 
               and example applications, just start them to see 
               irrKlang in action.
  \doc         Documentation of the Irrlicht Engine.
  \examples    Examples and tutorials showing how to use the native engine
               engine using C++.
  \example.net Examples written for the .NET version of irrKlang, in C# and
               VisualBasic.NET
  \include     Header files to include when using the engine.
  \lib         Lib to link with your programs when using the engine.
  \media       Sound data for the demo applications and
               examples.
  \plugins     Source and documentation of the irrKlang plugins.


==========================================================================
2. How to start
==========================================================================

  To see the engine in action in Windows, just go to the \bin\Win32
  directories, and start some applications. 
  
  To start developing own applications and games with the engine take 
  a look at the 01.HelloWorld example in the \examples directory.

  Windows Users: 
  There are Visual Studio as well as CodeBlocks projects for the examples.

  Linux users: 
  Just go into the /examples directory and run 'make' for every
  example you want to try out. You can run the examples directly from the
  directory they are created in. Some examples may print some special
  hints after a successful make and might help you starting them up 
  by typing in 'make run'.  
  
  Mac users: 
  There is a XCode project for each example available in 
  the corresponding example directory. To run the precompiled example
  applications from bin\macos-gcc, doubleclick the run_0x_xxxxx.command
  files from the finder (this will change the working directory to
  bin\macos-gcc) or start them directly from a terminal.
  	
==========================================================================
3. Release Notes
==========================================================================

  Informations about changes in this new version of the engine can be 
  found in changes.txt.

  Please note that the included sound files are copyright
  by their authors and not included in the irrKlang engine license. 


==========================================================================
4. .NET version dependencies
==========================================================================

There are four versions of irrKlang.NET available. They have the same 
functionality and can be used from the same .NET projects.
The difference are only their dependencies and target platforms:

irrKlang.NET2.0 in bin\dotnet-2.0:
      Compiled for .NET common language runtime version 2.0, depends on
      the modudles MSVCR80.DLL and MSVCM80.DLL. (32 bit)
	  One method to redistribute these is using the visual studio 2005
	  redistributeable (vcredist_x86)
	  
irrKlang.NET4 in bin\dotnet-4:
      Compiled for .NET common language runtime version 4.5, depends on
      VCRUNTIME140.dll. (32 bit)
	  One method to redistribute this is using the visual studio 2017
	  redistributeable (vcredist_x86)
      
irrKlang.NET4 in bin\dotnet-4-64: (only in the 64 bit irrKlang SDK)
	  64 bit version, Compiled for .NET common language runtime version 4.5,
	  depends on VCRUNTIME140.dll. (64 bit)
	  One method to redistribute this is using the visual studio 2017
	  redistributeable (vcredist 64 bit)
	
==========================================================================
5. License
==========================================================================

irrKlang's source codes, documentation and binaries contained within the 
distributed archive are copyright © Nikolaus Gebhardt / Ambiera 2001-2018.

The contents of the irrKlang distribution archive may not be redistributed, 
reproduced, modified, transmitted, broadcast, published or adapted in any 
way, shape or form, without the prior written consent of the owner, 
Nikolaus Gebhardt.

The irrKlang.dll, irrKlang.so and libirrklang.dylib files may be 
redistributed without the authors prior permission in non-commercial 
products, and must remain unmodified except for compressing the file.

For the included plugins which can be found in the \plugins
directory, different licenses may be applied, as specified in the 
actual plugin directory. 


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

--------------------------------------------------------------------------------
irrKlang may include Xiph.org Foundation code (Ogg Vorbis and FLAC).
It's license is the following:

Copyright (c) 2002, Xiph.org Foundation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of the Xiph.org Foundation nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

--------------------------------------------------------------------------------
irrKlang may include code from the
  Dynamic Universal Music Bibliotheque, Version 0.9.3

Copyright (C) 2001-2005 Ben Davis, Robert J Ohannessian and Julien Cugniere

==========================================================================
6. Contact
==========================================================================

  If you have problems, questions or suggestions, please visit the 
  official homepage of the irrKlang Engine:
  
  http://www.ambiera.com/irrklang
  
  You will find forums, bugtrackers, patches, tutorials, and other stuff
  which will help you out.
  
  If want to contact the author of the engine, please send an email to
  Nikolaus Gebhardt:
  
  office@ambiera.com
