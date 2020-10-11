
// MIT License
//
// Copyright (c) 2020 degski
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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <array>
#include <atomic>
#include <algorithm>
#include <sax/iostream.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <jthread>
#include <type_traits>
#include <utility>
#include <vector>

/*
    -fsanitize = address

    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan_cxx-x86_64.lib
    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan-preinit-x86_64.lib
    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan-x86_64.lib

    C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\lib\intel64_win\vc_mt\tbb.lib
*/

#include <sax/prng_sfc.hpp>
#include <sax/uniform_int_distribution.hpp>

#if defined( NDEBUG )
#    define RANDOM 1
#else
#    define RANDOM 0
#endif

namespace ThreadID {
// Creates a new ID.
[[nodiscard]] inline int get ( bool ) noexcept {
    static std::atomic<int> global_id = 0;
    return global_id++;
}
// Returns ID of this thread.
[[nodiscard]] inline int get ( ) noexcept {
    static thread_local int thread_local_id = get ( false );
    return thread_local_id;
}

} // namespace ThreadID

namespace Rng {
// Chris Doty-Humphrey's Small Fast Chaotic Prng.
[[nodiscard]] inline sax::Rng & generator ( ) noexcept {
    if constexpr ( RANDOM ) {
        static thread_local sax::Rng generator ( sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ) );
        return generator;
    }
    else {
        static thread_local sax::Rng generator ( sax::fixed_seed ( ) + ThreadID::get ( ) );
        return generator;
    }
}

} // namespace Rng

#undef RANDOM

sax::Rng & rng = Rng::generator ( );

inline std::uint32_t pointer_alignment ( const void * p_ ) noexcept {
    return ( uint32_t ) ( ( uintptr_t ) p_ & ( uintptr_t ) - ( ( intptr_t ) p_ ) );
}

#include "include/cascade_network.hpp"
#include "include/td_learning.hpp"
// __m128i _mm_minpos_epu16( __m128i packed_words);
int main ( ) { 
    
    cascade_network<2, 1, 3, 5> fccn ( rng ); 

    std::cout << fccn.NumWeights << nl;

    fccn.feed_forward ( );
    
    return EXIT_SUCCESS; 
}

#if 0

// Experimental Parameters:

int num_imputs, num_hidden, num_output;
int MAX_UNITS;         // maximum total number of units
int time_steps = 1024; // number of time steps to simulate
int time_step;         // no learning on time step 0

float bias;   // strength of the bias (constant input) contribution
float alpha;  // 1st layer learning rate (typically 1/num_inputs)
float beta;   // 2nd layer learning rate (typically 1/num_hidden)
float gamma;  // discount-rate parameter (typically 0.9)
float lambda; // trace decay parameter (should be <= gamma)

// Network Data Structure:

float l0[ time_steps ][ MAX_UNITS ]; // input data (units)
float l1[ MAX_UNITS ];               // hidden layer
float l2[ MAX_UNITS ];               // output layer
float w1[ MAX_UNITS ][ MAX_UNITS ];  // weights hidden layer
float w2[ MAX_UNITS ];               // weights output layer

// Learning Data Structure:

float old_y[ MAX_UNITS ];
float hidden_trace[ MAX_UNITS ][ MAX_UNITS ][ MAX_UNITS ];
float output_trace[ MAX_UNITS ][ MAX_UNITS ];
float reward[ time_steps ][ MAX_UNITS ];
float td_error[ MAX_UNITS ];

// Initialize weights and biases.
template<typename Generator>
void init_network ( Generator & ) noexcept;
// Compute hidden layer and output predictions.
void response ( ) noexcept;
// Update weight vectors.
void td_learn ( ) noexcept;
// Calculate new weight eligibilities.
void update_eligibilities ( ) noexcept;

// Initialize weights and biases.
template<typename Generator>
void init_network ( Generator & g_ ) noexcept {
    std::uniform_real_distribution<float> dis ( -1.0f + FLT_EPSILON, 1.0f - FLT_EPSILON );
    int i = 0;
    for ( int s = 0; s < time_steps; ++s )
        l0[ s ][ num_imputs ] = bias;
    l1[ num_hidden ] = bias;
    for ( int j = 0; j <= num_hidden; ++j ) {
        for ( int k = 0; k < num_output; ++k ) {
            w1[ j ][ k ]           = dis ( g_ );
            output_trace[ i ][ k ] = { };
            old_y[ k ]             = { };
        }
        for ( i = 0; i <= num_imputs; ++i ) {
            w2[ i ][ j ] = dis ( g_ );
            for ( int k = 0; k < num_output; ++k )
                hidden_trace[ i ][ j ][ k ] = { };
        }
    }
}

// Compute hidden layer and output predictions.
void response ( ) noexcept {
    l1[ num_hidden ]              = bias;
    l0[ time_step ][ num_imputs ] = bias;
    for ( int j = 0; j < num_hidden; ++j ) {
        l1[ j ] = { };
        for ( int i = 0; i <= num_imputs; ++i )
            l1[ j ] += l0[ time_step ][ i ] * w2[ i ][ j ];
        l1[ j ] = 1.0f / ( 1.0f + std::exp ( -l1[ j ] ) ); // asymmetric sigmoid
    }
    for ( int k = 0; k < num_output; ++k ) {
        l2[ k ] = { };
        for ( int j = 0; j <= num_hidden; ++j )
            l2[ k ] += l1[ j ] * w1[ j ][ k ];
        l2[ k ] = 1.0f / ( 1.0f + std::exp ( -l2[ k ] ) ); // asymmetric sigmoid (OPTIONAL)
    }
}

// Update weight vectors.
void td_learn ( ) noexcept {
    for ( int k = 0; k < num_output; ++k ) {
        for ( int j = 0; j <= num_hidden; ++j ) {
            w1[ j ][ k ] += beta * td_error[ k ] * output_trace[ j ][ k ];
            for ( int i = 0; i <= num_imputs; ++i )
                w2[ i ][ j ] += alpha * td_error[ k ] * hidden_trace[ i ][ j ][ k ];
        }
    }
}

// Calculate new weight eligibilities.
void update_eligibilities ( ) noexcept {
    float t[ MAX_UNITS ];
    for ( int k = 0; k < num_output; ++k )
        t[ k ] = l2[ k ] * ( 1.0f - l2[ k ] );
    for ( int j = 0; j <= num_hidden; ++j ) {
        for ( int k = 0; k < num_output; ++k ) {
            output_trace[ j ][ k ] = lambda * output_trace[ j ][ k ] + t[ k ] * l1[ j ];
            for ( int i = 0; i <= num_imputs; ++i )
                hidden_trace[ i ][ j ][ k ] = lambda * hidden_trace[ i ][ j ][ k ] +
                                              t[ k ] * w1[ j ][ k ] * l1[ j ] * ( 1.0f - l1[ j ] ) * l0[ time_step ][ i ];
        }
    }
}

int main ( ) {

    init_network ( rng );
    response ( ); // just compute old response (old_y)

    for ( int k = 0; k < num_output; ++k ) //
        old_y[ k ] = l2[ k ];              //

    update_eligibilities ( ); //...and prepare the eligibilities

    for ( time_step = 1; time_step <= time_steps; ++time_step ) {                    // a single pass through time series data
        response ( );                                                                // forward pass - compute activities
        for ( int k = 0; k < num_output; ++k )                                       //
            td_error[ k ] = reward[ time_step ][ k ] + gamma * l2[ k ] - old_y[ k ]; // form errors
        td_learn ( );                                                                // backward pass - learning
        response ( );                          // forward pass must be done twice to form TD errors
        for ( int k = 0; k < num_output; ++k ) //
            old_y[ k ] = l2[ k ];              // for use in next cycle's TD errors
        update_eligibilities ( );              // update eligibility traces
    }                                          // end time_step

    return EXIT_SUCCESS;
}

#endif
