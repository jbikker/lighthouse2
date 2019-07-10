#ifndef FREEIMAGE_H
#define FREEIMAGE_H

// Version information ------------------------------------------------------

#define FREEIMAGE_MAJOR_VERSION   3
#define FREEIMAGE_MINOR_VERSION   10
#define FREEIMAGE_RELEASE_SERIAL  0

// Compiler options ---------------------------------------------------------

#define DLL_CALLCONV __stdcall
#define DLL_API __declspec(dllimport)
#define FREEIMAGE_COLORORDER_BGR	0
#define FREEIMAGE_COLORORDER_RGB	1
#define FREEIMAGE_COLORORDER FREEIMAGE_COLORORDER_BGR
#define FI_DEFAULT(x)	= x
#define FI_ENUM(x)      enum x
#define FI_STRUCT(x)	struct x

// Bitmap types -------------------------------------------------------------

FI_STRUCT( FIBITMAP ) { void *data; };
FI_STRUCT( FIMULTIBITMAP ) { void *data; };

// Types used in the library (specific to FreeImage) ------------------------

#pragma pack(push, 1)

typedef struct tagFIRGB16 { WORD red, green, blue; } FIRGB16;
typedef struct tagFIRGBA16 { WORD red, green, blue, alpha; } FIRGBA16;
typedef struct tagFIRGBF { float red, green, blue; } FIRGBF;
typedef struct tagFIRGBAF { float red, green, blue, alpha; } FIRGBAF;
typedef struct tagFICOMPLEX { double r, i; } FICOMPLEX;

#pragma pack(pop)

#define FI_RGBA_RED				2
#define FI_RGBA_GREEN			1
#define FI_RGBA_BLUE			0
#define FI_RGBA_ALPHA			3
#define FI_RGBA_RED_MASK		0x00FF0000
#define FI_RGBA_GREEN_MASK		0x0000FF00
#define FI_RGBA_BLUE_MASK		0x000000FF
#define FI_RGBA_ALPHA_MASK		0xFF000000
#define FI_RGBA_RED_SHIFT		16
#define FI_RGBA_GREEN_SHIFT		8
#define FI_RGBA_BLUE_SHIFT		0
#define FI_RGBA_ALPHA_SHIFT		24
#define FI_RGBA_RGB_MASK		(FI_RGBA_RED_MASK|FI_RGBA_GREEN_MASK|FI_RGBA_BLUE_MASK)

// The 16bit macros only include masks and shifts, since each color element is not byte aligned

#define FI16_555_RED_MASK		0x7C00
#define FI16_555_GREEN_MASK		0x03E0
#define FI16_555_BLUE_MASK		0x001F
#define FI16_555_RED_SHIFT		10
#define FI16_555_GREEN_SHIFT	5
#define FI16_555_BLUE_SHIFT		0
#define FI16_565_RED_MASK		0xF800
#define FI16_565_GREEN_MASK		0x07E0
#define FI16_565_BLUE_MASK		0x001F
#define FI16_565_RED_SHIFT		11
#define FI16_565_GREEN_SHIFT	5
#define FI16_565_BLUE_SHIFT		0

// ICC profile support ------------------------------------------------------

#define FIICC_DEFAULT			0x00
#define FIICC_COLOR_IS_CMYK		0x01

FI_STRUCT( FIICCPROFILE )
{
	WORD    flags;	// info flag
	DWORD	size;	// profile's size measured in bytes
	void   *data;	// points to a block of contiguous memory containing the profile
};

// Important enums ----------------------------------------------------------

/** I/O image format identifiers.
*/
FI_ENUM( FREE_IMAGE_FORMAT )
{
	FIF_UNKNOWN = -1,
		FIF_BMP = 0,
		FIF_ICO = 1,
		FIF_JPEG = 2,
		FIF_JNG = 3,
		FIF_KOALA = 4,
		FIF_LBM = 5,
		FIF_IFF = FIF_LBM,
		FIF_MNG = 6,
		FIF_PBM = 7,
		FIF_PBMRAW = 8,
		FIF_PCD = 9,
		FIF_PCX = 10,
		FIF_PGM = 11,
		FIF_PGMRAW = 12,
		FIF_PNG = 13,
		FIF_PPM = 14,
		FIF_PPMRAW = 15,
		FIF_RAS = 16,
		FIF_TARGA = 17,
		FIF_TIFF = 18,
		FIF_WBMP = 19,
		FIF_PSD = 20,
		FIF_CUT = 21,
		FIF_XBM = 22,
		FIF_XPM = 23,
		FIF_DDS = 24,
		FIF_GIF = 25,
		FIF_HDR = 26,
		FIF_FAXG3 = 27,
		FIF_SGI = 28,
		FIF_EXR = 29,
		FIF_J2K = 30,
		FIF_JP2 = 31
};

/** Image type used in FreeImage.
*/
FI_ENUM( FREE_IMAGE_TYPE )
{
	FIT_UNKNOWN = 0,
		FIT_BITMAP = 1,
		FIT_UINT16 = 2,
		FIT_INT16 = 3,
		FIT_UINT32 = 4,
		FIT_INT32 = 5,
		FIT_FLOAT = 6,
		FIT_DOUBLE = 7,
		FIT_COMPLEX = 8,
		FIT_RGB16 = 9,
		FIT_RGBA16 = 10,
		FIT_RGBF = 11,
		FIT_RGBAF = 12
};

/** Image color type used in FreeImage.
*/
FI_ENUM( FREE_IMAGE_COLOR_TYPE )
{
	FIC_MINISWHITE = 0,
		FIC_MINISBLACK = 1,
		FIC_RGB = 2,
		FIC_PALETTE = 3,
		FIC_RGBALPHA = 4,
		FIC_CMYK = 5
};

/** Color quantization algorithms.
Constants used in FreeImage_ColorQuantize.
*/
FI_ENUM( FREE_IMAGE_QUANTIZE )
{
	FIQ_WUQUANT = 0,
		FIQ_NNQUANT = 1
};

/** Dithering algorithms.
Constants used in FreeImage_Dither.
*/
FI_ENUM( FREE_IMAGE_DITHER )
{
	FID_FS = 0,
		FID_BAYER4x4 = 1,
		FID_BAYER8x8 = 2,
		FID_CLUSTER6x6 = 3,
		FID_CLUSTER8x8 = 4,
		FID_CLUSTER16x16 = 5,
		FID_BAYER16x16 = 6
};

/** Lossless JPEG transformations
Constants used in FreeImage_JPEGTransform
*/
FI_ENUM( FREE_IMAGE_JPEG_OPERATION )
{
	FIJPEG_OP_NONE = 0,
		FIJPEG_OP_FLIP_H = 1,
		FIJPEG_OP_FLIP_V = 2,
		FIJPEG_OP_TRANSPOSE = 3,
		FIJPEG_OP_TRANSVERSE = 4,
		FIJPEG_OP_ROTATE_90 = 5,
		FIJPEG_OP_ROTATE_180 = 6,
		FIJPEG_OP_ROTATE_270 = 7
};

/** Tone mapping operators.
Constants used in FreeImage_ToneMapping.
*/
FI_ENUM( FREE_IMAGE_TMO )
{
	FITMO_DRAGO03 = 0,
		FITMO_REINHARD05 = 1,
		FITMO_FATTAL02 = 2
};

/** Upsampling / downsampling filters.
Constants used in FreeImage_Rescale.
*/
FI_ENUM( FREE_IMAGE_FILTER )
{
	FILTER_BOX = 0,
		FILTER_BICUBIC = 1,
		FILTER_BILINEAR = 2,
		FILTER_BSPLINE = 3,
		FILTER_CATMULLROM = 4,
		FILTER_LANCZOS3 = 5
};

/** Color channels.
Constants used in color manipulation routines.
*/
FI_ENUM( FREE_IMAGE_COLOR_CHANNEL )
{
	FICC_RGB = 0,
		FICC_RED = 1,
		FICC_GREEN = 2,
		FICC_BLUE = 3,
		FICC_ALPHA = 4,
		FICC_BLACK = 5,
		FICC_REAL = 6,
		FICC_IMAG = 7,
		FICC_MAG = 8,
		FICC_PHASE = 9
};

// Metadata support ---------------------------------------------------------

/**
  Tag data type information (based on TIFF specifications)

  Note: RATIONALs are the ratio of two 32-bit integer values.
*/
FI_ENUM( FREE_IMAGE_MDTYPE )
{
	FIDT_NOTYPE = 0,
		FIDT_BYTE = 1,
		FIDT_ASCII = 2,
		FIDT_SHORT = 3,
		FIDT_LONG = 4,
		FIDT_RATIONAL = 5,
		FIDT_SBYTE = 6,
		FIDT_UNDEFINED = 7,
		FIDT_SSHORT = 8,
		FIDT_SLONG = 9,
		FIDT_SRATIONAL = 10,
		FIDT_FLOAT = 11,
		FIDT_DOUBLE = 12,
		FIDT_IFD = 13,
		FIDT_PALETTE = 14
};

/**
  Handle to a metadata model
*/
FI_STRUCT( FIMETADATA ) { void *data; };

/**
  Handle to a FreeImage tag
*/
FI_STRUCT( FITAG ) { void *data; };

// File IO routines ---------------------------------------------------------

#ifndef FREEIMAGE_IO
#define FREEIMAGE_IO

typedef void* fi_handle;
typedef unsigned (DLL_CALLCONV *FI_ReadProc) (void *buffer, unsigned size, unsigned count, fi_handle handle);
typedef unsigned (DLL_CALLCONV *FI_WriteProc) (void *buffer, unsigned size, unsigned count, fi_handle handle);
typedef int (DLL_CALLCONV *FI_SeekProc) (fi_handle handle, long offset, int origin);
typedef long (DLL_CALLCONV *FI_TellProc) (fi_handle handle);

#if (defined(_WIN32) || defined(__WIN32__))
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif // WIN32

FI_STRUCT( FreeImageIO )
{
	FI_ReadProc  read_proc;     // pointer to the function used to read data
	FI_WriteProc write_proc;    // pointer to the function used to write data
	FI_SeekProc  seek_proc;     // pointer to the function used to seek
	FI_TellProc  tell_proc;     // pointer to the function used to aquire the current position
};

#if (defined(_WIN32) || defined(__WIN32__))
#pragma pack(pop)
#else
#pragma pack()
#endif // WIN32

/**
Handle to a memory I/O stream
*/
FI_STRUCT( FIMEMORY ) { void *data; };

#endif // FREEIMAGE_IO

// Plugin routines ----------------------------------------------------------

#ifndef PLUGINS
#define PLUGINS

typedef const char *(DLL_CALLCONV *FI_FormatProc) ();
typedef const char *(DLL_CALLCONV *FI_DescriptionProc) ();
typedef const char *(DLL_CALLCONV *FI_ExtensionListProc) ();
typedef const char *(DLL_CALLCONV *FI_RegExprProc) ();
typedef void *(DLL_CALLCONV *FI_OpenProc)(FreeImageIO *io, fi_handle handle, BOOL read);
typedef void (DLL_CALLCONV *FI_CloseProc)(FreeImageIO *io, fi_handle handle, void *data);
typedef int (DLL_CALLCONV *FI_PageCountProc)(FreeImageIO *io, fi_handle handle, void *data);
typedef int (DLL_CALLCONV *FI_PageCapabilityProc)(FreeImageIO *io, fi_handle handle, void *data);
typedef FIBITMAP *(DLL_CALLCONV *FI_LoadProc)(FreeImageIO *io, fi_handle handle, int page, int flags, void *data);
typedef BOOL( DLL_CALLCONV *FI_SaveProc )(FreeImageIO *io, FIBITMAP *dib, fi_handle handle, int page, int flags, void *data);
typedef BOOL( DLL_CALLCONV *FI_ValidateProc )(FreeImageIO *io, fi_handle handle);
typedef const char *(DLL_CALLCONV *FI_MimeProc) ();
typedef BOOL( DLL_CALLCONV *FI_SupportsExportBPPProc )(int bpp);
typedef BOOL( DLL_CALLCONV *FI_SupportsExportTypeProc )(FREE_IMAGE_TYPE type);
typedef BOOL( DLL_CALLCONV *FI_SupportsICCProfilesProc )();

FI_STRUCT( Plugin )
{
	FI_FormatProc format_proc;
	FI_DescriptionProc description_proc;
	FI_ExtensionListProc extension_proc;
	FI_RegExprProc regexpr_proc;
	FI_OpenProc open_proc;
	FI_CloseProc close_proc;
	FI_PageCountProc pagecount_proc;
	FI_PageCapabilityProc pagecapability_proc;
	FI_LoadProc load_proc;
	FI_SaveProc save_proc;
	FI_ValidateProc validate_proc;
	FI_MimeProc mime_proc;
	FI_SupportsExportBPPProc supports_export_bpp_proc;
	FI_SupportsExportTypeProc supports_export_type_proc;
	FI_SupportsICCProfilesProc supports_icc_profiles_proc;
};

typedef void (DLL_CALLCONV *FI_InitProc)(Plugin *plugin, int format_id);

#endif // PLUGINS


// Load / Save flag constants -----------------------------------------------

#define BMP_DEFAULT         0
#define BMP_SAVE_RLE        1
#define CUT_DEFAULT         0
#define DDS_DEFAULT			0
#define EXR_DEFAULT			0		// save data as half with piz-based wavelet compression
#define EXR_FLOAT			0x0001	// save data as float instead of as half (not recommended)
#define EXR_NONE			0x0002	// save with no compression
#define EXR_ZIP				0x0004	// save with zlib compression, in blocks of 16 scan lines
#define EXR_PIZ				0x0008	// save with piz-based wavelet compression
#define EXR_PXR24			0x0010	// save with lossy 24-bit float compression
#define EXR_B44				0x0020	// save with lossy 44% float compression - goes to 22% when combined with EXR_LC
#define EXR_LC				0x0040	// save images with one luminance and two chroma channels, rather than as RGB (lossy compression)
#define FAXG3_DEFAULT		0
#define GIF_DEFAULT			0
#define GIF_LOAD256			1		// Load the image as a 256 color image with ununsed palette entries, if it's 16 or 2 color
#define GIF_PLAYBACK		2		// 'Play' the GIF to generate each frame (as 32bpp) instead of returning raw frame data when loading
#define HDR_DEFAULT			0
#define ICO_DEFAULT         0
#define ICO_MAKEALPHA		1		// convert to 32bpp and create an alpha channel from the AND-mask when loading
#define IFF_DEFAULT         0
#define J2K_DEFAULT			0		// save with a 16:1 rate
#define JP2_DEFAULT			0		// save with a 16:1 rate
#define JPEG_DEFAULT        0		// loading (see JPEG_FAST); saving (see JPEG_QUALITYGOOD)
#define JPEG_FAST           0x0001	// load the file as fast as possible, sacrificing some quality
#define JPEG_ACCURATE       0x0002	// load the file with the best quality, sacrificing some speed
#define JPEG_CMYK			0x0004	// load separated CMYK "as is" (use | to combine with other load flags)
#define JPEG_QUALITYSUPERB  0x80	// save with superb quality (100:1)
#define JPEG_QUALITYGOOD    0x0100	// save with good quality (75:1)
#define JPEG_QUALITYNORMAL  0x0200	// save with normal quality (50:1)
#define JPEG_QUALITYAVERAGE 0x0400	// save with average quality (25:1)
#define JPEG_QUALITYBAD     0x0800	// save with bad quality (10:1)
#define JPEG_PROGRESSIVE	0x2000	// save as a progressive-JPEG (use | to combine with other save flags)
#define KOALA_DEFAULT       0
#define LBM_DEFAULT         0
#define MNG_DEFAULT         0
#define PCD_DEFAULT         0
#define PCD_BASE            1		// load the bitmap sized 768 x 512
#define PCD_BASEDIV4        2		// load the bitmap sized 384 x 256
#define PCD_BASEDIV16       3		// load the bitmap sized 192 x 128
#define PCX_DEFAULT         0
#define PNG_DEFAULT         0
#define PNG_IGNOREGAMMA		1		// avoid gamma correction
#define PNM_DEFAULT         0
#define PNM_SAVE_RAW        0       // If set the writer saves in RAW format (i.e. P4, P5 or P6)
#define PNM_SAVE_ASCII      1       // If set the writer saves in ASCII format (i.e. P1, P2 or P3)
#define PSD_DEFAULT         0
#define RAS_DEFAULT         0
#define SGI_DEFAULT			0
#define TARGA_DEFAULT       0
#define TARGA_LOAD_RGB888   1       // If set the loader converts RGB555 and ARGB8888 -> RGB888.
#define TIFF_DEFAULT        0
#define TIFF_CMYK			0x0001	// reads/stores tags for separated CMYK (use | to combine with compression flags)
#define TIFF_PACKBITS       0x0100  // save using PACKBITS compression
#define TIFF_DEFLATE        0x0200  // save using DEFLATE compression (a.k.a. ZLIB compression)
#define TIFF_ADOBE_DEFLATE  0x0400  // save using ADOBE DEFLATE compression
#define TIFF_NONE           0x0800  // save without any compression
#define TIFF_CCITTFAX3		0x1000  // save using CCITT Group 3 fax encoding
#define TIFF_CCITTFAX4		0x2000  // save using CCITT Group 4 fax encoding
#define TIFF_LZW			0x4000	// save using LZW compression
#define TIFF_JPEG			0x8000	// save using JPEG compression
#define WBMP_DEFAULT        0
#define XBM_DEFAULT			0
#define XPM_DEFAULT			0


#ifdef __cplusplus
extern "C" {
#endif

	DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFileType( const char *filename, int size FI_DEFAULT( 0 ) );
	DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFIFFromFilename( const char *filename );
	DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Load( FREE_IMAGE_FORMAT fif, const char *filename, int flags FI_DEFAULT( 0 ) );
	DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertTo24Bits( FIBITMAP *dib );
	DLL_API unsigned DLL_CALLCONV FreeImage_GetWidth( FIBITMAP *dib );
	DLL_API unsigned DLL_CALLCONV FreeImage_GetHeight( FIBITMAP *dib );
	DLL_API BYTE *DLL_CALLCONV FreeImage_GetScanLine( FIBITMAP *dib, int scanline );
	DLL_API void DLL_CALLCONV FreeImage_Unload( FIBITMAP *dib );
	DLL_API BOOL DLL_CALLCONV FreeImage_GetHistogram( FIBITMAP *dib, DWORD *histo, FREE_IMAGE_COLOR_CHANNEL channel FI_DEFAULT( FICC_BLACK ) );
	DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertTo32Bits( FIBITMAP *dib );
	DLL_API unsigned DLL_CALLCONV FreeImage_GetPitch( FIBITMAP *dib );
	DLL_API unsigned DLL_CALLCONV FreeImage_GetBPP( FIBITMAP *dib );
	DLL_API BOOL DLL_CALLCONV FreeImage_Invert( FIBITMAP *dib );
	DLL_API FIBITMAP *DLL_CALLCONV FreeImage_GetChannel( FIBITMAP *dib, FREE_IMAGE_COLOR_CHANNEL channel );
	DLL_API BYTE *DLL_CALLCONV FreeImage_GetBits( FIBITMAP *dib );

	// restore the borland-specific enum size option
#if defined(__BORLANDC__)
#pragma option pop
#endif

#ifdef __cplusplus
}
#endif

#endif // FREEIMAGE_H