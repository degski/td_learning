
// this file was lifted from the SimdNet repo on 01.06.2020 and will be further
// developed here.

// MIT License
//
// Copyright (c) 2019, 2020 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "td_learning/detail/simd_exp.inl"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <mkl.h>

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <sax/iostream.hpp>
#include <span>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>

using span_ps       = std::span<float>;
using const_span_ps = std::span<float const>;

inline constexpr float euler_constant_ps = 2.718'281'746f;

// a back-of-the-envelope-calc
template<int NumInput, int NumOutput, int NumNeurons>
class scratch_space;

template<int NumInput, int NumOnes, int NumOutput, int NumNeurons>
struct cascade_network;

namespace calc {

inline constexpr int roundup_multiple ( int i_, int m_ ) noexcept { return ( ( i_ + m_ - 1 ) / m_ ) * m_; }

namespace detail {

template<int NumInput, int NumOnes, int NumOutput, int NumNeurons, int Padding>
class alignas ( 32 ) aligned_storage { // every array can be cache aligned - blas-strides != 1

    template<int, int, int>
    friend class scratch_space;
    template<int, int, int>
    friend class cascade_network;

    std::array<float, Padding> _ = { };
    std::array<float, NumInput> raw;
    std::array<float, NumOnes> one;
    std::array<float, NumNeurons - NumOutput> hid;
    std::array<float, roundup_multiple ( NumOutput, 32 / sizeof ( float ) )> out; // out is AVX ready
};

template<int NumInput, int NumOnes, int NumOutput, int NumNeurons>
class alignas ( 32 ) aligned_storage<NumInput, NumOnes, NumOutput, NumNeurons, 0> {

    template<int, int, int>
    friend class scratch_space;
    template<int, int, int>
    friend class cascade_network;

    std::array<float, NumInput> raw;
    std::array<float, NumOnes> one;
    std::array<float, NumNeurons - NumOutput> hid;
    std::array<float, roundup_multiple ( NumOutput, 32 / sizeof ( float ) )> out; // out is AVX ready
};

} // namespace detail

template<int NumInput, int NumOnes, int NumOutput, int NumNeurons>
class space {

    template<int, int, int>
    friend class scratch_space;
    template<int, int, int>
    friend class cascade_network;

    void zero ( ) noexcept {
        storage.raw.fill ( 0.0f );
        storage.one.fill ( 1.0f );
    }

    detail::aligned_storage<NumInput, NumOnes, NumOutput, NumNeurons,
                            roundup_multiple ( ( NumInput + NumOnes + NumNeurons ) - NumOutput, 32 / sizeof ( float ) ) -
                                ( ( NumInput + NumOnes + NumNeurons ) - NumOutput )>
        storage;
};

template<typename T>
struct reverse_container_wrapper {
    T & reverse_iterable;
};

template<typename T>
auto begin ( reverse_container_wrapper<T> w ) {
    return std::rbegin ( w.reverse_iterable );
}

template<typename T>
auto end ( reverse_container_wrapper<T> w ) {
    return std::rend ( w.reverse_iterable );
}

template<typename T>
reverse_container_wrapper<T> reverse_container_adaptor ( T && reverse_iterable_ ) {
    return { reverse_iterable_ };
}

// a back-of-the-envelope-calc
template<int NumInput, int NumOnes, int NumOutput, int NumNeurons>
class scratch_space {

    template<int, int, int>
    friend class cascade_network;

    using space = space<NumInput, NumOnes, NumOutput, NumNeurons>;

    static_assert ( NumNeurons >= NumOutput, "number of neurons needs to be equal or larger than the number of required outputs" );

    // zone-end-index

    static constexpr int NumInp       = NumInput + NumOnes;
    static constexpr int NumInpHid    = NumInput + NumOnes + NumNeurons - NumOutput;
    static constexpr int NumInpHidOut = NumInput + NumOnes + NumNeurons;

    static constexpr int NumWeights = ( NumNeurons * ( 2 * NumInpHidOut ) ) / 2;

    public:
    scratch_space ( ) noexcept = default;

    void clear_scratch_space ( ) noexcept { scratch.zero ( ); }
    [[nodiscard]] static constexpr int size ( ) noexcept { return NumInput + 1 + NumNeurons; }

    [[nodiscard]] static constexpr int weight_size ( ) noexcept { return NumWeights; }
    [[nodiscard]] static constexpr int neuron_size ( ) noexcept { return NumNeurons; }

    [[nodiscard]] constexpr float & operator[] ( int i ) noexcept { return scratch[ i ]; }
    [[nodiscard]] constexpr float const & operator[] ( int i ) const noexcept { return scratch[ i ]; }

    [[nodiscard]] constexpr float * data ( ) noexcept { return scratch.storage.raw ( ); }
    [[nodiscard]] constexpr float const * data ( ) const noexcept { return scratch.storage.raw ( ); }

    // includes bias
    [[nodiscard]] span_ps inp ( ) noexcept { return { scratch.storage.raw ( ), NumInput + 1 }; }
    [[nodiscard]] const_span_ps inp ( ) const noexcept { return { scratch.storage.raw ( ), NumInput + 1 }; }
    // excludes bias
    [[nodiscard]] auto & raw ( ) noexcept { return scratch.storage.raw; }
    [[nodiscard]] auto const & raw ( ) const noexcept { return scratch.storage.raw; }

    [[nodiscard]] auto & hid ( ) noexcept { return scratch.storage.hid; }
    [[nodiscard]] auto const & hid ( ) const noexcept { return scratch.storage.hid; }

    [[nodiscard]] span_ps out ( ) noexcept { return { scratch.storage.out ( ), NumOutput }; }
    [[nodiscard]] const_span_ps out ( ) const noexcept { return { scratch.storage.out ( ), NumOutput }; }

    [[nodiscard]] span_ps neu ( ) noexcept { return { scratch.hid.scratch ( ), NumNeurons }; }
    [[nodiscard]] const_span_ps neu ( ) const noexcept { return { scratch.hid.scratch ( ), NumNeurons }; }

    [[nodiscard]] span_ps all ( ) noexcept { return { scratch.storage.raw ( ), NumInpHidOut }; }
    [[nodiscard]] const_span_ps all ( ) const noexcept { return { scratch.storage.raw ( ), NumInpHidOut }; }

    template<typename Stream>
    [[maybe_unused]] friend Stream & operator<< ( Stream & out_, scratch_space const & space_ ) noexcept {
        for ( auto const s : space_.all ( ) )
            out_ << s << ' ';
        out_ << nl;
        return out_;
    }

    private:
    space scratch;
};

} // namespace calc

// cascade_network
//
//   In a 'cascade_network' all up-stream neurons are input to all down-stream neurons. The last neuron receives input from all
//   neurons (incl. itself), all biases, and all raw-inputs. In this model, all neurons are a single neuron in its respective 'own'
//   layer.
//
template<int NumInput, int NumOnes, int NumOutput, int NumNeurons>
struct cascade_network {

    static_assert ( NumNeurons >= NumOutput, "number of neurons needs to be equal or larger than the number of required outputs" );

    // zone-end-index

    static constexpr int NumInp       = NumInput + NumOnes;
    static constexpr int NumInpHid    = NumInput + NumOnes + NumNeurons - NumOutput;
    static constexpr int NumInpHidOut = NumInput + NumOnes + NumNeurons;

    static constexpr int NumWeights = ( NumNeurons * ( 2 * NumInpHidOut ) ) / 2;

    static constexpr float alpha = 0.25f; // learning

    using wgt_type = std::array<float, NumWeights>;
    using out_type = std::array<float, NumOutput>;

    using pointer        = typename wgt_type::pointer;
    using const_pointer  = typename wgt_type::const_pointer;
    using iterator       = typename wgt_type::iterator;
    using const_iterator = typename wgt_type::const_iterator;

    template<typename Generator>
    cascade_network ( Generator & rng_ ) noexcept {
        std::uniform_real_distribution<float> dis ( -1.0f + FLT_EPSILON, 1.0f - FLT_EPSILON ); // closed interval
        std::generate ( std::begin ( weights ), std::end ( weights ), [ &rng_, &dis ] ( ) noexcept { return dis ( rng_ ); } );
    }

    void feed_forward_soft_max ( ) noexcept {
        float max = 0.0f, sum = 0.0f;
        for ( auto & o : space.out ( ) ) {
            if ( o > max )
                max = o;
            sum += ( o = normalized_exponential_function_activation ( o ) );
        }
        for ( auto & o : space.out ( ) )
            ( ( o /= sum ) += max );
    }

    void feed_forward ( ) const noexcept {
        auto dat = space.data ( ), wgt = weights.data ( );
        int i = NumInp;
        for ( auto & n : space.neu ( ) ) {
            n = rectifier_activation ( cblas_sdot ( i, dat, 1, wgt, 1 ) * alpha );
            wgt += i++;
        }
    }

    // returns sum absolute error
    [[nodiscard]] float feed_backward ( out_type const & desired_activation_ ) const noexcept {
        float e   = 0.0f;
        pointer p = &*std::next ( space.scratch.storage.out.rbegin ( ) ), d = &*desired_activation_.rbegin ( );

        for ( auto & o : reverse_container_adaptor ( space.out ( ) ) )
            e += std::abs ( ( *p-- += derivative_normalized_exponential_function_activation ( o - *d-- ) ) );

        for ( auto & h : reverse_container_adaptor ( space.hid ( ) ) )
            *p-- += derivative_rectifier_activation ( h );

        return e;
    }

    // https://godbolt.org/z/QWtr96

    [[nodiscard]] float fabs_branchless ( float f_ ) const noexcept {
        unsigned int u;
        memcpy ( &u, &f_, sizeof ( unsigned int ) );
        u &= ~( 1ul << 31 );
        memcpy ( &f_, &u, sizeof ( unsigned int ) );
        return f_;
    }

    float fabs_branchless_alt ( float f_ ) noexcept {
        unsigned long x;
        memcpy ( &x, &f_, sizeof ( unsigned long ) );
        unsigned long m = x >> 31;
        m               = x + m ^ m;
        memcpy ( &f_, &m, sizeof ( unsigned long ) );
        return f_;
    }

    [[nodiscard]] float elliotsig_activation ( float net_alpha_ ) const noexcept {
        net_alpha_ /= 1.0f + std::abs ( net_alpha_ ); // branchless after optimization
        return net_alpha_;
    }
    [[nodiscard]] float derivative_elliotsig_activation ( float elliotsig_activation_ ) const noexcept {
        elliotsig_activation_ *= elliotsig_activation_;
        return elliotsig_activation_;
    }

    [[nodiscard]] float parametric_rectifier_activation ( float net_alpha_, float rectifier_alpha_ ) const noexcept {
        int n = 0;
        std::memcpy ( &n, &net_alpha_, 1 );
        n >>= 31;
        net_alpha_ *= ( float ) n + std::forward<float> ( rectifier_alpha_ ) * ( float ) not( ( bool ) n );
        return net_alpha_;
    }

    [[nodiscard]] float rectifier_activation ( float net_alpha_ ) const noexcept {
        return parametric_rectifier_activation ( std::forward<float> ( net_alpha_ ), 0.00f );
    }

    [[nodiscard]] float leaky_rectifier_activation ( float net_alpha_ ) const noexcept {
        return parametric_rectifier_activation ( std::forward<float> ( net_alpha_ ), 0.01f );
    }

    [[nodiscard]] float normalized_exponential_function_activation ( float net_alpha_ ) const noexcept {
        net_alpha_ = std::powf ( euler_constant_ps, net_alpha_ );
        return net_alpha_;
    }

    [[nodiscard]] float derivative_activation ( float activation_ ) const noexcept {
        unsigned long n = 0ul;
        std::memcpy ( &n, &activation_, 1 );
        n >>= 31;
        activation_ = ( float ) n;
        return activation_;
    }

    [[nodiscard]] float & operator[] ( int i_ ) noexcept { return weights[ i_ ]; }
    [[nodiscard]] float const & operator[] ( int i_ ) const noexcept { return weights[ i_ ]; }

    [[nodiscard]] constexpr pointer data ( ) noexcept { return weights.data ( ); }
    [[nodiscard]] constexpr const_pointer * data ( ) const noexcept { return weights.data ( ); }

    [[nodiscard]] iterator begin ( ) noexcept { return iterator ( weights.begin ( ) ); }
    [[nodiscard]] iterator end ( ) noexcept { return iterator ( weights.end ( ) ); }
    [[nodiscard]] const_iterator begin ( ) const noexcept { return const_iterator ( weights.begin ( ) ); }
    [[nodiscard]] const_iterator cbegin ( ) const noexcept { return const_iterator ( weights.cbegin ( ) ); }
    [[nodiscard]] const_iterator end ( ) const noexcept { return const_iterator ( weights.end ( ) ); }
    [[nodiscard]] const_iterator cend ( ) const noexcept { return const_iterator ( weights.cend ( ) ); }

    // the second operator+=() embodies 'the log-sum-exp trick' (as opposed to operator=()).

    // endian test.
    template<typename ValueType>
    static constexpr int hi_byte_index ( ) noexcept {
        short s = 1;
        return char ( s ) * ( sizeof ( ValueType ) - 1 );
    }

    static constexpr bool is_little_endian = hi_byte_index<short> ( );

    template<typename Stream>
    [[maybe_unused]] friend Stream & operator<< ( Stream & out_, cascade_network const & w_ ) noexcept {
        for ( auto const v : w_.weights )
            out_ << v << ' ';
        out_ << nl;
        return out_;
    }

    friend class cereal::access;

    template<class Archive>
    void serialize ( Archive & ar_ ) {
        ar_ ( weights );
    }

    alignas ( 64 ) static thread_local calc::scratch_space<NumInput, NumOnes, NumOutput,
                                                           NumNeurons> space; // input-bias-hidden-output - scratch space

    wgt_type weights;
};
