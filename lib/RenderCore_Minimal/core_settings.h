/* core_settings.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   The settings and classes in this file are core-specific:
   - avilable in host and device code
   - specific to this particular core.
   Global settings can be configured shared.h.
*/

#pragma once

#include "platform.h"

#ifdef _DEBUG
#pragma comment(lib, "../platform/lib/debug/platform.lib" )
#else
#pragma comment(lib, "../platform/lib/release/platform.lib" )
#endif

using namespace lighthouse2;

#include "core_api_base.h"
#include "core_api.h"
#include "rendercore.h"

using namespace lh2core;

// EOF