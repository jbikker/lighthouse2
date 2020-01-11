/**
 * Helper utilities to provide details about object storage,
 * such as required size and alignment for objects from a list of types.
 */

#pragma once

// TODO: Check how windows nvrtc responds to this!
#ifdef _MSC_VER
#define ALIGN( x ) __declspec( align( x ) )
#else
#define ALIGN( x ) __attribute__( ( aligned( x ) ) )
#endif

// https://developercommunity.visualstudio.com/content/problem/560274/error-c2988-unrecognizable-template-declarationdef.html
// https://stackoverflow.com/questions/53705903/stdis-convertibleargtypes-validtraitsvalue-dependent-name-is-not-a-type
// Reportedly fixed in 2019 16.3
#if defined(_MSC_VER) && _MSC_VER < 1923

template <bool... v>
struct all_true
{
	static constexpr bool value = std::min( {v...} );
};

template <bool... v>
struct any_true
{
	static constexpr bool value = std::max( {v...} );
};

#else /* MSVC BUG */

template <bool... v>
using all_true = std::integral_constant<bool, std::min( {v...} )>;

template <bool... v>
using any_true = std::integral_constant<bool, std::max( {v...} )>;

#endif /* MSVC BUG */

template <typename same, typename... more>
using any_is_same = any_true<std::is_same<same, more>::value...>;

template <typename to, typename... from>
using all_convertible_to = all_true<std::is_convertible<from, to>::value...>;

template <typename... Ts>
struct StorageDetail
{
	static constexpr size_t alignment = std::max( {alignof( Ts )...} );
	static constexpr size_t size = std::max( {sizeof( Ts )...} );
};

template <typename... Ts>
struct StorageRequirement
{
	using Detail = StorageDetail<Ts...>;

	static constexpr auto stack_alignment = Detail::alignment;
	static constexpr auto stack_element_size = Detail::size;
	static_assert( stack_alignment >= 1, "Storage must have an alignment of at least 1!" );
	static_assert( stack_element_size >= 1, "Storage must have a size of at least 1!" );

	// TODO: https://stackoverflow.com/a/15912208/2844473 Doesn't seem to work with using syntax...
	// TODO: LH2 is C++11... Use alignas everywhere instead of a macro based on the platform.
	// using type ALIGN( stack_alignment ) = char[stack_element_size];
	// using type alignas( stack_alignment ) = char[stack_element_size];

	/**
	 * An abstract type that is large enough to store any of the types in Ts, with the right right storage requirements
	 */
	typedef ALIGN( stack_alignment ) char type[stack_element_size];
	// Note that "warning: alignas does not apply here" is thrown by this,
	// but the assert below is not triggered and the code functions correctly.
	// typedef char type alignas( stack_alignment )[stack_element_size];

	static_assert( sizeof( char ) == 1, "Char must be 1 byte in size!" );
	static_assert( alignof( type ) == stack_alignment, "Array storage type lost its alignment!" );

	template <typename T>
	static constexpr bool HasType()
	{
		return any_is_same<T, Ts...>::value;
	}
};
