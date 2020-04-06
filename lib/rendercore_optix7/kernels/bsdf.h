#ifndef BSDF_H
#define BSDF_H

#include "noerrors.h"
#include "compatibility.h"

#if 0

// simple reference bsdf: Lambert plus specular reflection
#include "lambert.h"

#else

// Disney's principled BRDF, adapted from AppleSeed
#include "ggxmdf.h"
#include "frosted.h"
#if 1
#include "disney.h"
#else
#include "disney_ref.h"
#endif

#endif

#endif // BSDF_H

// EOF