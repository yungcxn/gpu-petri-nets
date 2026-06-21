#ifndef COMMON_H
#define COMMON_H

#include <assert.h>
#include <stdint.h>
#include <math.h>

#define CONCATENATE(a, b) CONCATENATE_INNER(a, b)
#define CONCATENATE_INNER(a, b) a ## b

#define MAX_UINT_FOR_TYPE(TYPE) ((TYPE)(~(TYPE)0))
#define ROUND_UP(n, pad) (((n) + ((pad) - 1)) & ~((pad) - 1))
#define ROUNDF_UP(n, pad) (ceilf((n) / (pad)) * (pad))

#define ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))
#define SAFE_FREE(ptr) do { free((ptr)); (ptr) = NULL; } while (0)
#define SAFE_CUDAFREE(ptr) do { cudaFree((ptr)); (ptr) = NULL; } while (0)
#define SAFE_CUDAFREEHOST(ptr) do { cudaFreeHost((ptr)); (ptr) = NULL; } while (0)

#ifdef DEBUG
  #define DEBUG_PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
  #define DEBUG_PRINTF(...) ((void)0)
#endif

static uint32_t next_power_of_two(uint32_t n) {
  uint32_t p = 1;
  if (n && !(n & (n - 1))) return n;
  while (p < n) p <<= 1;
  return p;
}

#define ASSERT(x)                           \
  do {                                      \
    if (!(x)) {                             \
      printf("ASSERT-ERROR: ");             \
      printf(#x);                           \
      printf("\n");                         \
      exit(EXIT_FAILURE);                   \
    }                                       \
  } while(0)

#define DEV_ASSERT(x)                      \
  do {                                      \
    if (!(x)) {                             \
      printf("ASSERT-ERROR: ");             \
      printf(#x);                           \
      printf("\n");                         \
      assert(false);                        \
    }                                       \
  } while(0)

#define CUDA_ERR_ASSERT()                                  \
  do {                                                     \
    cudaError_t err = cudaGetLastError();                  \
    if (err != cudaSuccess) {                              \
      printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); \
      assert(false);                                       \
  }                                                        \
  } while(0)

typedef uint8_t bit;

template<typename T, typename G>
struct pair_t {
  T a;
  G b;
};

template <typename A, typename B>
struct is_same {
  static constexpr bool value = false;
};

template <typename A>
struct is_same<A, A> {
  static constexpr bool value = true;
};

template<bool b, typename T, typename F>
struct _cond_type {
  using type = F;
};

template<typename T, typename F>
struct _cond_type<true, T, F> {
  using type = T;
};

template<bool b, typename T, typename F>
using cond_type = typename _cond_type<b, T, F>::type;

#define SAME_TYPE(X,Y) is_same<X,Y>::value
#define COND_TYPE(B,T,F) cond_type<B,T,F>

#define CONSTEXPR_ASSIGN(be, var, a, b) \
  if constexpr (be) var = a; \
  else var = b;

#endif 