
// this code was lifted from SO (license).

#define USE_FMA true

/* max. rel. error = 1.72863156e-3 on [-87.33654, 88.72283] */
[[nodiscard]] __m128 __mm_exp_ps ( __m128 x ) noexcept { // https://stackoverflow.com/a/47025627/646940
    __m128 t, f, e, p, r;
    __m128i i, j;
    __m128 l2e = _mm_set1_ps ( 1.442695041f ); /* log2(e) */
    __m128 c0  = _mm_set1_ps ( 0.3371894346f );
    __m128 c1  = _mm_set1_ps ( 0.657636276f );
    __m128 c2  = _mm_set1_ps ( 1.00172476f );

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    t = _mm_mul_ps ( x, l2e ); /* t = log2(e) * x */
#ifdef __SSE4_1__
    e = _mm_floor_ps ( t );                                               /* floor(t) */
    i = _mm_cvtps_epi32 ( e );                                            /* (int)floor(t) */
#else                                                                     /* __SSE4_1__*/
    i = _mm_cvttps_epi32 ( t );                        /* i = (int)t */
    j = _mm_srli_epi32 ( _mm_castps_si128 ( x ), 31 ); /* signbit(t) */
    i = _mm_sub_epi32 ( i, j );                        /* (int)t - signbit(t) */
    e = _mm_cvtepi32_ps ( i );                         /* floor(t) ~= (int)t - signbit(t) */
#endif                                                                    /* __SSE4_1__*/
    f = _mm_sub_ps ( t, e );                                              /* f = t - floor(t) */
    p = c0;                                                               /* c0 */
    p = _mm_mul_ps ( p, f );                                              /* c0 * f */
    p = _mm_add_ps ( p, c1 );                                             /* c0 * f + c1 */
    p = _mm_mul_ps ( p, f );                                              /* (c0 * f + c1) * f */
    p = _mm_add_ps ( p, c2 );                                             /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    j = _mm_slli_epi32 ( i, 23 );                                         /* i << 23 */
    r = _mm_castsi128_ps ( _mm_add_epi32 ( j, _mm_castps_si128 ( p ) ) ); /* r = p * 2^i*/
    return r;
}

/* compute exp(x) for x in [-87.33654f, 88.72283]
   maximum relative error: 3.1575e-6 (USE_FMA = 0); 3.1533e-6 (USE_FMA = 1)
*/
[[nodiscard]] __m256 _mm256_exp_ps ( __m256 x ) { // https://stackoverflow.com/a/49090523/646940
    __m256 t, f, p, r;
    __m256i i, j;

    const __m256 l2e = _mm256_set1_ps ( 1.442695041f );    /* log2(e) */
    const __m256 l2h = _mm256_set1_ps ( -6.93145752e-1f ); /* -log(2)_hi */
    const __m256 l2l = _mm256_set1_ps ( -1.42860677e-6f ); /* -log(2)_lo */
    /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
    const __m256 c0 = _mm256_set1_ps ( 0.041944388f );
    const __m256 c1 = _mm256_set1_ps ( 0.168006673f );
    const __m256 c2 = _mm256_set1_ps ( 0.499999940f );
    const __m256 c3 = _mm256_set1_ps ( 0.999956906f );
    const __m256 c4 = _mm256_set1_ps ( 0.999999642f );

    /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
    t = _mm256_mul_ps ( x, l2e );                                             /* t = log2(e) * x */
    r = _mm256_round_ps ( t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); /* r = rint (t) */

#if USE_FMA
    f = _mm256_fmadd_ps ( r, l2h, x ); /* x - log(2)_hi * r */
    f = _mm256_fmadd_ps ( r, l2l, f ); /* f = x - log(2)_hi * r - log(2)_lo * r */
#else                                  // USE_FMA
    p = _mm256_mul_ps ( r, l2h );                      /* log(2)_hi * r */
    f = _mm256_add_ps ( x, p );                        /* x - log(2)_hi * r */
    p = _mm256_mul_ps ( r, l2l );                      /* log(2)_lo * r */
    f = _mm256_add_ps ( f, p );                        /* f = x - log(2)_hi * r - log(2)_lo * r */
#endif                                 // USE_FMA

    i = _mm256_cvtps_epi32 ( t ); /* i = (int)rint(t) */

    /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
    p = c0; /* c0 */
#if USE_FMA
    p = _mm256_fmadd_ps ( p, f, c1 ); /* c0*f+c1 */
    p = _mm256_fmadd_ps ( p, f, c2 ); /* (c0*f+c1)*f+c2 */
    p = _mm256_fmadd_ps ( p, f, c3 ); /* ((c0*f+c1)*f+c2)*f+c3 */
    p = _mm256_fmadd_ps ( p, f, c4 ); /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
#else                                 // USE_FMA
    p = _mm256_mul_ps ( p, f );                        /* c0*f */
    p = _mm256_add_ps ( p, c1 );                       /* c0*f+c1 */
    p = _mm256_mul_ps ( p, f );                        /* (c0*f+c1)*f */
    p = _mm256_add_ps ( p, c2 );                       /* (c0*f+c1)*f+c2 */
    p = _mm256_mul_ps ( p, f );                        /* ((c0*f+c1)*f+c2)*f */
    p = _mm256_add_ps ( p, c3 );                       /* ((c0*f+c1)*f+c2)*f+c3 */
    p = _mm256_mul_ps ( p, f );                        /* (((c0*f+c1)*f+c2)*f+c3)*f */
    p = _mm256_add_ps ( p, c4 );                       /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
#endif                                // USE_FMA

    /* exp(x) = 2^i * p */
    j = _mm256_slli_epi32 ( i, 23 );                                               /* i << 23 */
    r = _mm256_castsi256_ps ( _mm256_add_epi32 ( j, _mm256_castps_si256 ( p ) ) ); /* r = p * 2^i */

    return r;
}

/* if higher accuracy is required, the degree of the polynomial approximation can be bumped up by one, using the following set of
    coefficients:
   maximum relative error: 1.7428e-7 (USE_FMA = 0); 1.6586e-7 (USE_FMA = 1)
const __m256 c0 = _mm256_set1_ps ( 0.008301110f );
const __m256 c1 = _mm256_set1_ps ( 0.041906696f );
const __m256 c2 = _mm256_set1_ps ( 0.166674897f );
const __m256 c3 = _mm256_set1_ps ( 0.499990642f );
const __m256 c4 = _mm256_set1_ps ( 0.999999762f );
const __m256 c5 = _mm256_set1_ps ( 1.000000000f );
*/
