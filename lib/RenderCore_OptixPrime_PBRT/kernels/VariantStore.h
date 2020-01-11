#pragma once

// #include <assert.h>
#define assert(a)
#include <utility>

template <typename BaseType, typename Storage>
class other_type_iterator
{
	Storage ptr;

  public:
	typedef BaseType value_type;
	typedef size_t difference_type;
	typedef value_type& reference;
	typedef value_type* pointer;

	__device__ other_type_iterator( Storage ptr ) : ptr( ptr )
	{
	}

	__device__ reference operator*() const
	{
		return *reinterpret_cast<pointer>( ptr );
	}

	__device__ pointer operator->() const
	{
		return reinterpret_cast<pointer>( ptr );
	}

	__device__ other_type_iterator& operator++()
	{
		ptr++;
		return *this;
	}

	__device__ other_type_iterator& operator--()
	{
		ptr--;
		return *this;
	}

	__device__ bool operator==( const other_type_iterator& other ) const
	{
		return ptr == other.ptr;
	}

	__device__ bool operator!=( const other_type_iterator& other ) const
	{
		return ptr != other.ptr;
	}
};

/**
 * Stack storage for all Variants, that are all subclasses
 * of Base.
 */
template <typename Base, typename... Variants>
class VariantStore
{
	// Check if every type is a Base. They are casted as such from a typeless stack,
	// so make sure the cast/reinterpretation is valid.
	static_assert( all_convertible_to<Base&, Variants&...>::value,
				   "One or more Variants are not a subclass of the Base type!" );

	using Req = StorageRequirement<Variants...>;
	using StorageType = typename Req::type;

	// static constexpr auto max_elements = 8;
	static constexpr auto max_elements = sizeof...(Variants);

	// For some reason the alignment of the type is not propagated
	// to the use here, despite the static_assert not failing.
	static_assert( alignof( StorageType ) == Req::stack_alignment,
				   "StorageType lost required alignment!" );
	alignas( Req::stack_alignment ) StorageType stack[max_elements];
	size_t items = 0;

	// Disable warnings about dropping type attributes
	// While this doesn't contribute towards correctness,
	// it works like it should right now.
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

	typedef other_type_iterator<Base, StorageType*> iterator;
	typedef other_type_iterator<const Base, const StorageType*> const_iterator;

#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic pop
#endif

  public:
	__device__ size_t size() const
	{
		return items;
	}

	__device__ iterator begin()
	{
		return stack;
	}

	__device__ const const_iterator begin() const
	{
		return stack;
	}

	__device__ iterator end()
	{
		return stack + items;
	}

	__device__ const const_iterator end() const
	{
		return stack + items;
	}

	__device__ Base& operator[]( size_t idx )
	{
		assert( idx < items );
		return *reinterpret_cast<Base*>( stack + idx );
	}

	__device__ const Base& operator[]( size_t idx ) const
	{
		assert( idx < items );
		return *reinterpret_cast<const Base*>( stack + idx );
	}

	// Pass type to know what constructor to invoke
	template <typename T, typename... Args>
	__device__ std::decay_t<T>& emplace_back( Args&&... args )
	{
		// Decay template argument into the underlying type:
		// (This removes const qualifiers and references)
		using simple_type = std::decay_t<T>;

		// Make sure the added type fits on the stack
		static_assert( any_is_same<simple_type, Variants...>::value,
					   "Type does not fit on the stack!" );

		// Construct in place
		return *new ( Reserve() ) simple_type( std::forward<Args>( args )... );
	}

	// Overload for emplacing an object on the stack, invoking the
	// copy/move constructor. This is a helper to resolve T
	// as the first "Args" type
	template <typename T>
	__device__ std::decay_t<T>& emplace_back( T&& arg )
	{
		return emplace_back<T, T>( std::forward<T>( arg ) );
	}

	template <typename T>
	__device__ void push_back( T&& arg )
	{
		emplace_back<T>( std::forward<T>( arg ) );
	}

  private:
	__device__ void* Reserve()
	{
		assert( items < max_elements );
		return stack + items++;
	}
};

__device__ static void compile_time_tests()
{
	struct Base
	{
		float a;
		__device__ Base( float a ) : a( a ) {}
	};

	struct Thing : public Base
	{
		int x;
		__device__ Thing( float a, int x ) : Base( a ), x( x ) {}
	};

	struct Thing2 : public Base
	{
		__device__ Thing2( float a ) : Base( a ) {}
	};

	VariantStore<Base, Thing, Thing2> store;

	store.emplace_back<Thing>( 1.f, 1 );
	store.push_back( Thing( 1.f, 1 ) );
	auto nonconst = Thing( 1.f, 1 );
	store.push_back( nonconst );

	// The variant store accepts multiple types by definition. Invoking
	// emplace_back with constructor arguments requires specifying the
	// desired type to construct:
	store.emplace_back<Thing>( 2.f, 3 );
	// Test if an rvalue can be "emplaced" into the list
	// (effectively invoking the move constructor)
	store.emplace_back( Thing2( 3.f ) );
	store.emplace_back<Thing2>( 3.f );

	auto& ref1 = store.emplace_back( nonconst );
	auto& ref2 = store.emplace_back( std::move( nonconst ) );
	const auto cnst = Thing( 1.f, 1 );
	store.push_back( cnst );
	auto& ref3 = store.emplace_back( cnst );
	auto& ref4 = store.emplace_back( std::move( cnst ) );
	static_assert( !std::is_const<std::remove_reference_t<decltype( ref3 )>>::value,
				   "Returned refernce for const-emplace_back must not be const!" );
}
