/* author: Can Nayci */

/* COMPILATION FLAGS:            */
/*                               */
/* - DEBUG      (for prints)     */
/* - JAVA_SUPPORT                */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <device_launch_parameters.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include "headers/common.cuh"
#include "headers/fileloaderv2.cuh"

/******************************************** simulation mode specification ***/

#define LOCK_FREE 0
#define LOCK_WRITING 1
#define LOCK_WRITTEN 2

#define NET_NAME_BE      "BE-Net"
#define NET_NAME_PT      "PT-Net"
#define NET_NAME_KK      "KK-Net"
#define NET_NAME_MINIBE  "Mini-BE-Net"

/* since pt_transition_elem_t is uint2, upper 32bit is weight, lower placeid */
#define PT_TRANSITION_ELEM_GET_WEIGHT(X) X.x
#define PT_TRANSITION_ELEM_GET_PLACE(X) X.y 

/* look below into the typedefs to understand the marking array layout for kk */
#define KK_PLACE_GET_SUBELEM_PTR(m_a, p_id, c_id, glob_color_degree) \
  (&(m_a)[(p_id) * (glob_color_degree) + (c_id)])

/* be place is either marked (0xFF) or unmarked (0x00)                        */
#define BE_MARKED              MAX_UINT_FOR_TYPE(be_place_t)
/* the AoSoA transition data has padded inner arrays, and pad is:             */
#define BE_TRANSITION_LINE_END MAX_UINT_FOR_TYPE(be_transition_elem_t)
#define PT_TRANSITION_LINE_END MAX_UINT_FOR_TYPE(pt_transition_elem_placepart_t)
#define KK_TRANSITION_LINE_END MAX_UINT_FOR_TYPE(kk_transition_subelem_t)
#define MINIBE_TRANSITION_LINE_END MAX_UINT_FOR_TYPE(minibe_transitionchunk_t)

#define RG_NODEARR_END         0xFFFFFFFF

#define TRANSITION_PADDING      128 /* bytes */
#define PLACE_PADDING           128 /* bytes */
#define METADATA_PADDING        128 /* bytes */
#define TRANSITION_AREA_PADDING 512 /* this is important for texture creation */
                                    /* bytes */   

#define METADATA_FIELD_UNSET 0xFFFFFFFF
#define MAGIC_NUMBER_OFFSET  31

/********************************************************************* misc ***/

#define MURMUR_C1 0x87c37b91114253d5ULL
#define MURMUR_C2 0x4cf5ad432745937fULL
#define MURMUR_C3 0x52dce729
#define MURMUR_C4 0xff51afd7ed558ccdULL
#define MURMUR_C5 0xc4ceb9fe1a85ec53ULL

/* if on linux */
#if defined(__linux__)
  #define CLOCK_TYPE CLOCK_MONOTONIC_RAW /* used for exact time measurements */
#else  
  #define CLOCK_TYPE CLOCK_MONOTONIC
#endif

/* for magic number of file format */
enum _net_type_id {
  NET_TYPE_BE =      0,
  NET_TYPE_PT =      1,
  NET_TYPE_KK =      2,
  NET_TYPE_MINIBE = 99 /* special type, not directly in magic number slot */
};

#define NET_SPECIALFIELD_MINIBE_INDICATOR ((uint32_t)7) /* one field before   */
                                                        /* real magic num     */

/******************************************** simulation mode specification ***/

/* marking array is here [FF, 00, 00, FF, 00, ...] consisting of single bytes */
/* transition data is AoSoA,                                                  */
/*   - array of transitions                                                   */
/*   - each transition consists of two lines, a preset and a postset line     */
/*   - one "line" (preset and postset) consists of uint32 indices             */
/*   - indices point to places to consume the token (pre) / produce to (post) */
typedef uint8_t  be_place_t;
typedef uint32_t be_transition_elem_t;

/* marking array [0x00000001, 0x00000010, ...] of uint32 token counters       */
/* transition data like be, but pre/postset line consists of uint64s          */
/*   each uint64 consists of <weight, placeindex> to consume/produce tokens   */
typedef int32_t  pt_place_t;
typedef uint2    pt_transition_elem_t;
typedef uint32_t pt_transition_elem_placepart_t;
typedef int32_t  pt_transition_elem_weightpart_t;

/* marking array consists of n*uint32 token counters, where n is the          */
/*   maximum colour domain size across all places, since all marking types    */
/*   in one colour are mapped to 0...n-1. (n * kk_place_subelem_t's = 1 place)*/
/* transition data is like pt, but pre/postset line consists of (n+1)*uint32s */
/*   which looks like <placeindex, weight_markingtype_1, _2, _3, ..., _n>     */
/*   ((n+1) * kk_transition_subelem_t = 1 transition element)                 */
/* -> so this is technically an AoSoAoS                                       */
typedef int32_t kk_place_subelem_t;
typedef uint32_t kk_transition_subelem_t;

/* marking array is like be but in binary format.                             */
/* transition is padded like every other format, but in masked format         */
/* meaning preset-line has place-amount of bits with 1s for places to consume */
/* and postset-line has place-amount of bits with 1s for places to produce    */
typedef uint64_t minibe_placechunk_t;
typedef uint64_t minibe_transitionchunk_t;

/************************************************************* custom types ***/

typedef struct {
  cudaTextureObject_t* arr;
  uint32_t             amount;
  uint32_t             min_transitions_per_tex;
} fragm_transitions_tex_t;

/* gathered through CUDA API, flexible across devices */
typedef struct {
  uint32_t max_threads_per_block;
  uint32_t max_threads_per_sm;
  uint32_t max_concurrent_blocks;
  uint32_t max_regs_per_sm;
  uint32_t max_regs_per_thread;
  uint32_t sms;
  uint32_t cuda_cores;
  uint32_t l1_per_core;
  uint32_t l2;
  uint32_t max_shared_per_block;
  uint32_t max_concurrent_threads;
  uint32_t max_texture_1d_linear;
} device_prop_t;

typedef struct {
  void* dev_marking_arr;
  void* dev_transitions_arr;
} void_net_t;

typedef struct {
  be_place_t* dev_marking_arr;
  be_transition_elem_t* dev_transitions_arr;
} be_net_t;

typedef struct {
  pt_place_t* dev_marking_arr;
  pt_transition_elem_t* dev_transitions_arr;
} pt_net_t;

typedef struct {
  kk_place_subelem_t* dev_marking_arr;
  kk_transition_subelem_t* dev_transitions_arr;
} kk_net_t;

typedef struct {
  minibe_placechunk_t* dev_marking_arr;
  minibe_transitionchunk_t* dev_transitions_arr;
} minibe_net_t;

/***************************************************************** RG types ***/

/* 128 byte - cacheline aligned - arc */
typedef struct __attribute__((packed)) {
  uint32_t from; /* index in list of nodes */
  uint32_t to;
  uint32_t transition_id;
  uint32_t padding;
} rg_arc_t;

typedef struct {
  uint32_t places;
  uint32_t transitions;
  uint32_t transition_line_elems;
  uint32_t glob_color_degree;
  uint32_t magic_number;

  void*     transition_arr;

  void*     dev_nodes_arr;      /* is of place type, and padded */ 
  uint32_t  bytes_per_node;
  uint32_t  maxnodes;
  rg_arc_t* dev_arcs_arr;
  uint32_t  maxarcs;
} rg_data_t;

typedef struct {
  uint32_t* front_now;
  uint32_t* front_next;
  int32_t   head_now;
  int32_t   head_next;
} frontier_t;

/************************************************************* helper funcs ***/

static void print_info(device_prop_t* dev_props) {
  printf("Requiring net format for .cbe:\n");
  printf("  - 4 bytes marking count\n");
  printf("  - 4 bytes transition count\n");
  printf("  - 4 bytes transition line elements\n");
  printf("  - 128-4*3 bytes padding\n");
  printf("  - after these 128 bytes: marking-bytes (128-byte padded) \n");
  printf("  - after marking bytes: transition area (128-byte padded)\n");
  printf("  - one transition is preset+postset-line\n");
  printf("  - each line is padded to 128 bytes\n");
  printf("  - less transitions than or equal to %d\n",
         dev_props->max_concurrent_threads);
  printf("  - transition line elements must be a multiple of %lu\n",
         TRANSITION_PADDING / sizeof(be_transition_elem_t));
  printf("  - transition area byte-size must be a multiple of %d\n", 
         TRANSITION_AREA_PADDING);
  printf("  - transition area must begin at pointer-multiple of %d\n", 
         TRANSITION_AREA_PADDING);

  printf("\nRequiring net format for .cpt:\n");
  printf("  - 4 bytes marking count\n");
  printf("  - 4 bytes transition count\n");
  printf("  - 4 bytes transition line elements\n");
  printf("  - 128-4*4 bytes padding\n");
  printf("  - 4 bytes magic number, must be 1 for this net-type\n");
  printf("  - after: marking-array (signed int32s, 128-byte padded) \n");
  printf("  - after marking bytes: transition area (128-byte padded)\n");
  printf("  - one transition is preset+postset-line\n");
  printf("  - one transition line has uint64 elements\n");
  printf("    - first 32bit: arc-weight \n");
  printf("    - last 32bit: place-index \n");
  printf("  - each line is padded to 128 bytes\n");
  printf("  - less transitions than or equal to %d\n",
         dev_props->max_concurrent_threads);
  printf("  - transition line elements must be a multiple of %lu\n",
         TRANSITION_PADDING / sizeof(pt_transition_elem_t));
  printf("  - transition area byte-size must be a multiple of %d\n", 
         TRANSITION_AREA_PADDING);
  printf("  - transition area must begin at pointer-multiple of %d\n", 
         TRANSITION_AREA_PADDING);

  printf("\nRequiring net format for .ckk:\n");
  printf("  - 4 bytes marking count\n");
  printf("  - 4 bytes transition count\n");
  printf("  - 4 bytes transition line elements\n");
  printf("  - 128-4*4 bytes padding\n");
  printf("  - 4 bytes magic number, must be 2 for this net-type\n");
  printf("  - after: marking-array (uint32s, 128-byte padded) \n");
  printf("  - after marking bytes: transition area (128-byte padded)\n");
  printf("  - one transition is preset+postset-line\n");
  printf("  - one transition line has (n+1)*uint32 elements\n");
  printf("    - first 32bit: place-index \n");
  printf("    - next n*32bit: arc-weights for each token type\n");
  printf("  - each line is padded to 128 bytes\n");
  printf("  - less transitions than or equal to %d\n",
         dev_props->max_concurrent_threads);
  printf("  - transition line elements must be a multiple of %lu\n",
         TRANSITION_PADDING / sizeof(kk_transition_subelem_t));
  printf("  - transition area byte-size must be a multiple of %d\n", 
         TRANSITION_AREA_PADDING);
  printf("  - transition area must begin at pointer-multiple of %d\n", 
         TRANSITION_AREA_PADDING);
}

static void print_dev_info(device_prop_t* dev_props) {
  printf("Device Properties:\n");
  printf("  Max Threads per Block: %u\n", dev_props->max_threads_per_block);
  printf("  Max Threads per SM: %u\n", dev_props->max_threads_per_sm);
  printf("  Max Concurrent Blocks: %u\n", dev_props->max_concurrent_blocks);
  printf("  Max Registers per SM: %u\n", dev_props->max_regs_per_sm);
  printf("  Max Registers per Thread: %u\n", dev_props->max_regs_per_thread);
  printf("  SMs: %u\n", dev_props->sms);
  printf("  CUDA Cores: %u\n", dev_props->cuda_cores);
  printf("  L1 Cache per Core: %u bytes\n", dev_props->l1_per_core);
  printf("  L2 Cache: %u bytes\n", dev_props->l2);
  printf("  Max Shared Memory per Block: %u bytes\n",
         dev_props->max_shared_per_block);
  printf("  Max Texture 1D Linear Size: %u bytes\n", 
          dev_props->max_texture_1d_linear);
}

static void print_usage(const char* progname) {
  printf(
    "Usage:\n"
    "  %s <steps/0> <file-in> <file/pipe-out>\n"
    "  cat file_in | %s <nodes> <file/pipe-out>\n"
    "  %s --info/-i               # Print device info\n"
    "  %s --device/-d             # Print detailed device info\n"
    "  %s --help/-h               # Print this help message\n\n"
    "Arguments:\n"
    "  nodes            Number of nodes (0 for interactive mode)\n"
    "  file-in          Input file or stdin pipe\n"
    "  file/pipe-out    Output file or pipe\n"
    "\nExamples:\n"
    "  %s 100 input.cpt out\n"
    "  %s 11 input.cbe out\n",
    progname, progname, progname, progname, progname, progname, progname
  );
}

/**
 * convertSMVer2Cores - Convert SM version to number of cores
 *
 * @major: Major version number 
 * @minor: Minor version number
 *
 * This function converts the SM version (major and minor) to the number
 * of CUDA cores per SM for various NVIDIA GPU architectures.
 *
 * Return: Number of CUDA cores per SM
 */
static uint32_t convertSMVer2Cores(uint32_t major, uint32_t minor) {
  typedef struct {
    uint32_t SM;  /* 0xMm (major * 0x10 + minor) */
    uint32_t cores;
  } sm_to_cores_t;

  sm_to_cores_t arch_cores_per_sm[] = {
    {0x30, 192}, /* Kepler                */
    {0x32, 192}, /* Kepler                */
    {0x35, 192}, /* Kepler                */
    {0x37, 192}, /* Kepler                */
    {0x50, 128}, /* Maxwell               */
    {0x52, 128}, /* Maxwell               */
    {0x53, 128}, /* Maxwell               */
    {0x60,  64}, /* Pascal                */
    {0x61, 128}, /* Pascal                */
    {0x62, 128}, /* Pascal                */
    {0x70,  64}, /* Volta                 */
    {0x72,  64}, /* Xavier                */
    {0x75,  64}, /* Turing                */
    {0x80,  64}, /* Ampere                */
    {0x86, 128}, /* Ampere (GA10x)        */
    {0x87, 128}, /* Ampere (Jetson Orin)  */
    {0x89, 128}, /* Ada Lovelace (L4)     */
    {0x90, 128}, /* Hopper (GH100)        */
    {0xA0, 128}, /* Blackwell (estimated) */
    {0x0,  0x0}
  };

  uint32_t sm_version = (major << 4) + minor;
  for (uint32_t i = 0; arch_cores_per_sm[i].SM != 0; ++i) {
    if (arch_cores_per_sm[i].SM == sm_version) {
      return arch_cores_per_sm[i].cores;
    }
  }

  printf("Warning: Unknown SM version %d.%d, assuming 128 cores.\n",
         major, minor);
  return 128;
}


/**
 * get_device_properties - Get device properties
 * 
 * @device_id: ID of the CUDA device (default is 0)
 *
 * This function retrieves various properties of the specified CUDA device
 * and returns them in a device_prop_t structure.
 * 
 * Return: device_prop_t structure containing device properties
 */
static device_prop_t get_device_properties(uint32_t device_id = 0) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  device_prop_t dev_props;
  dev_props.max_threads_per_block     = prop.maxThreadsPerBlock;
  dev_props.max_threads_per_sm        = prop.maxThreadsPerMultiProcessor;
  dev_props.max_regs_per_sm           = prop.regsPerMultiprocessor;
  dev_props.max_shared_per_block      = prop.sharedMemPerBlock;
  dev_props.l1_per_core               = prop.sharedMemPerMultiprocessor;
  dev_props.l2                        = prop.l2CacheSize;
  dev_props.sms                       = prop.multiProcessorCount;
  uint32_t cores_per_sm               = convertSMVer2Cores(prop.major, 
                                                           prop.minor);
  dev_props.cuda_cores                = dev_props.sms * cores_per_sm;
  dev_props.max_regs_per_thread       = prop.regsPerBlock 
                                        / prop.maxThreadsPerBlock;
  uint32_t assumed_threads_per_block  = prop.maxThreadsPerBlock;
  dev_props.max_concurrent_blocks     = dev_props.sms *
                                        (dev_props.max_threads_per_sm 
                                         / assumed_threads_per_block);
  dev_props.max_concurrent_threads    = dev_props.sms
                                        * dev_props.max_threads_per_sm;
  dev_props.max_texture_1d_linear     = prop.maxTexture1D;
  return dev_props;
}


static double timespec_diff_ms(struct timespec *start, struct timespec *end) {
  return (end->tv_sec - start->tv_sec) * 1000.0 
         + (end->tv_nsec - start->tv_nsec) / 1e6;
}


__device__ __forceinline__ int int_to_str(char *buf, int val) {
  int len = 0;
  if (val == 0) {
    buf[0] = '0';
    return 1;
  }
  char tmp[10];
  while (val > 0) {
    tmp[len++] = '0' + (val % 10);
    val /= 10;
  }
  for (int i = 0; i < len; ++i) {
    buf[i] = tmp[len - 1 - i];
  }
  return len;
}

/*************************************************************** rg helpers ***/

/**
 * arc_publish - Publish an arc to the arc array  
 * @arcs: Pointer to the arc array
 * @from: Source node index in the node array
 * @to: Target node index in the node array
 * @tid: Transition ID, which will be written as arc info aswell
 * @queue_tail: Pointer to the atomic counter for the next free arc index
 * @max: Maximum number of arcs in the arc array
 * 
 * This function atomically publishes an arc to the arc array by
 * incrementing the queue tail counter and writing the arc data
 * to the corresponding index in the arc array.
 *
 * Return: void
 */
__device__ __forceinline__ void arc_publish(rg_arc_t* arcs, uint32_t from,
                                            uint32_t to, uint32_t tid,
                                            uint32_t* queue_tail, uint32_t max){

  uint32_t arc_idx = atomicAdd(queue_tail, 1);
  if (arc_idx < max) {
    rg_arc_t* arc = &arcs[arc_idx];
    arc->from = from;
    arc->to = to;
    arc->transition_id = tid;
    arc->padding = 0;
  }
}

/* helper func for murmur hash */
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

/** 
 * warp_murmur3_hash - warp-distributed Murmur3 hash
 * @data: Pointer to the data to be hashed
 * @uint64s: Number of uint64 elements in the data 
 * @lane: Lane ID of the thread in the warp
 *
 * This function computes the Murmur3 hash of the given data using
 * warp-level parallelism. Each thread in the warp processes a portion
 * of the data, and the results are combined to produce the final hash.
 *
 * Context: Does not expect fully active warp
 * Return: Computed Murmur3 hash value, the same for all lanes
 */
__device__ __forceinline__ uint64_t warp_murmur3_hash(const uint64_t* data,
                                                      const uint32_t uint64s,
                                                      const uint32_t lane) {
  uint64_t h = 0; /* seed */

  for (uint32_t i = lane; i < uint64s; i += 32) {
    uint64_t k = data[i];
    k *= MURMUR_C1;
    k = rotl64(k, 31);
    k *= MURMUR_C2;

    h ^= k;
    h = rotl64(h, 27);
    h = h * 5 + MURMUR_C3;
  }

  /* mix lane to break symmetry in xor reduction */
  uint64_t k_lane = lane;
  k_lane *= MURMUR_C1; k_lane = rotl64(k_lane, 31); k_lane *= MURMUR_C2;
  h ^= k_lane;

  unsigned mask = __activemask();
  for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
    h ^= __shfl_xor_sync(mask, h, offset);
  }

  /* finalizer */
  h ^= (uint64s * 8);
  h ^= h >> 33;
  h *= MURMUR_C4;
  h ^= h >> 33;
  h *= MURMUR_C5;
  h ^= h >> 33;

  return h;
}

/**
 * frontier_switch - Switch frontiers for next iteration
 * @dev_f: Pointer to the frontier structure on device 
 * 
 * This function switches the current and next frontiers in the frontier
 * structure, preparing for the next iteration of processing.
 *
 * Return: void
 */ 
__device__ __forceinline__ void frontier_switch(frontier_t* dev_f) {
  dev_f->head_now = dev_f->head_next;
  uint32_t* tmp = dev_f->front_now;
  dev_f->front_now = dev_f->front_next;
  dev_f->front_next = tmp;
  dev_f->head_next = 0;
  __threadfence();
}

/**
 * frontier_push_next - Push a value to the next frontier
 * @dev_f: Pointer to the frontier structure on device
 * @val: Value to be pushed to the next frontier
 *
 * This function atomically pushes a value to the next frontier
 * in the frontier structure and returns the index where the value
 * was inserted.
 *
 * Return: Index where the value was inserted in the next frontier
 */
__device__ __forceinline__ int32_t frontier_push_next(frontier_t* dev_f,
                                                      uint32_t val) {
  int32_t head_next = atomicAdd(&(dev_f->head_next), 1);
  dev_f->front_next[head_next] = val;
  return head_next;
}

/**
 * frontier_devcreate - Create a frontier structure on device
 * @maxnodes: Maximum number of nodes in the frontier
 * 
 * This function allocates and initializes a frontier structure
 * on the device with the specified maximum number of nodes.
 *
 * Return: Pointer to the created frontier structure on device
 */
static inline frontier_t* frontier_devcreate(uint32_t maxnodes) {
  
  frontier_t* f;
  cudaMalloc(&f, sizeof(frontier_t));
  uint32_t* front1;
  uint32_t* front2;
  cudaMalloc(&front1, maxnodes * sizeof(uint32_t));
  cudaMalloc(&front2, maxnodes * sizeof(uint32_t));

  /* front1 has 0 as first item */
  cudaMemset(front1, 0, sizeof(uint32_t));

  frontier_t h_f;
  h_f.front_now = front1;
  h_f.front_next = front2;
  h_f.head_now = 0;
  h_f.head_next = 0;

  cudaMemcpy(f, &h_f, sizeof(frontier_t), cudaMemcpyHostToDevice);
  return f;
}

/**
 * frontier_devdestroy - Destroy a frontier structure on device
 * @f: Pointer to the frontier structure on device
 * 
 * This function frees the memory allocated for the frontier
 * structure and its associated frontiers on the device.
 *
 * Return: void
 */
static inline void frontier_devdestroy(frontier_t* f) {
  frontier_t h_f;
  cudaMemcpy(&h_f, f, sizeof(frontier_t), cudaMemcpyDeviceToHost);
  cudaFree(h_f.front_now);
  cudaFree(h_f.front_next);
  cudaFree(f);
}

/**
 * warp_aligned_copy - Copy data between warp-aligned buffers
 * @dst: Destination buffer
 * @src: Source buffer
 * @uint64s_per_n: Number of uint64_t elements per node
 * @lane: Thread lane index
 *
 * This function performs a warp-aligned copy of data from the source
 * buffer to the destination buffer for the specified thread lane.
 *
 * Return: void
 */
__device__ __forceinline__ void warp_aligned_copy(uint64_t* dst, 
                                                  const uint64_t* src, 
                                                  const uint32_t uint64s_per_n,
                                                  const uint32_t lane) {

  for (uint32_t i = lane; i < uint64s_per_n; i += 32) {
    dst[i] = src[i];
  }

}

/**
  * warp_arrays_equal - Check if two arrays are equal using warp parallelism
 * @a: First array
 * @b: Second array
 * @count: Number of uint64_t elements in each array
 * @lane: Thread lane index
 *
 * This function checks if two arrays are equal by comparing their elements
 * using warp-level parallelism. Each thread in the warp compares a portion
 * of the arrays, and the results are combined to determine if the arrays
 * are equal.
 *
 * Context: Does not expect fully active warp
 *
 * Return: true if arrays are equal, false otherwise
 */
__device__ __forceinline__ bool warp_arrays_equal(const uint64_t* a,
                                                  const uint64_t* b,
                                                  const uint32_t count,
                                                  const uint32_t lane) {

  bool is_equal = true; 

  for (uint32_t i = lane; i < count; i += 32) {
    if (a[i] != b[i]) { 
      is_equal = false; 
    }
  }
  
  return __all_sync(__activemask(), is_equal);
}

/**
 * glob_syncd_frontier_switch - Global barrier synchronization across with
 *                              frontier switch
 * @dev_sync_data: Pointer to the device synchronization data array
 * @total_threads: Total number of threads participating in the sync
 * @local_sense: Local sense variable for the calling thread
 * @dev_f: Pointer to the frontier structure on device
 *
 * This function implements a global barrier synchronization mechanism
 * using atomic operations and a sense-reversing technique. It ensures
 * that all threads reach the barrier before any of them proceed.
 *
 * It is needed as a cooperative lib replacement, as its prone to deadlocks,
 * aswell as needing a special launch configuration.
 *
 * Return: void
 */
__device__ __forceinline__ 
void glob_syncd_frontier_switch(uint32_t* dev_sync_data, 
                               const uint32_t total_threads,
                               uint32_t& local_sense,
                               frontier_t* dev_f) {

  __threadfence();
  if (atomicAdd(&dev_sync_data[0], 1) == total_threads - 1) {
    DEBUG_PRINTF("tid-%d: Releasing global sync\n", 
                 threadIdx.x + blockIdx.x * blockDim.x);

    frontier_switch(dev_f);
    /* last thread to arrive */
    atomicExch(&dev_sync_data[0], 0);
    __threadfence();
    atomicExch(&dev_sync_data[1], local_sense);
  } else {
    DEBUG_PRINTF("tid-%d: Waiting at global sync\n",
                 threadIdx.x + blockIdx.x * blockDim.x);
    while (atomicAdd(&dev_sync_data[1], 0) != local_sense) {
      /* spin-wait */
    }
  }
  local_sense = !local_sense;
}

/* -------------------------------------------------------------------------- */
/* ------------------------------- main part -------------------------------- */
/* -------------------------------- device ---------------------------------- */
/* -------------------------------------------------------------------------- */

/** 
 * pt_warp_try_fire - Check if a warp can fire a transition
 * @warp_marking_buf: Buffer holding the marking for each place
 * @warp_tl_pre: Pre-transition elements for the warp
 * @warp_tl_post: Post-transition elements for the warp
 * @transition_line_elems: Number of elements in the transition line
 *
 * This function checks if a warp can fire a transition by examining the
 * marking of the places involved in the transition. If all places have
 * sufficient tokens, the transition can fire.
 *
 * Context: Expects warp without manually disabled lanes. 
 *          Also leaves behind a corrupted warp_marking_buf if not firable.
 *
 * Return: true if the transition can fire, false otherwise
 */
__device__ __forceinline__ 
bool pt_warp_try_fire(pt_place_t* warp_marking_buf, 
                      const pt_transition_elem_t* warp_tl_pre,
                      const pt_transition_elem_t* warp_tl_post,
                      const uint32_t transition_line_elems, 
                      const uint32_t lane) {

  bool firable = true;
  for (uint32_t i = lane; i < transition_line_elems; i += 32) {
    pt_transition_elem_t transition_elem = __ldg(&warp_tl_pre[i]);

    pt_transition_elem_placepart_t placepart =
      PT_TRANSITION_ELEM_GET_PLACE(transition_elem); /* get place */
    pt_transition_elem_weightpart_t weightpart = 
      PT_TRANSITION_ELEM_GET_WEIGHT(transition_elem); /* get weight */

    if (placepart != PT_TRANSITION_LINE_END) { /* skip end marker */
      warp_marking_buf[placepart] -= weightpart; /* decrement buffer */
      if (warp_marking_buf[placepart] < 0) {
        firable = false;
      }
    }
  }

  __syncwarp();
  if (!__all_sync(__activemask(), firable)) return false;
  
  for (uint32_t i = lane; i < transition_line_elems; i += 32) {
    pt_transition_elem_t transition_elem = __ldg(&warp_tl_post[i]);

    pt_transition_elem_placepart_t placepart = 
      PT_TRANSITION_ELEM_GET_PLACE(transition_elem);

    pt_transition_elem_weightpart_t weightpart = 
      PT_TRANSITION_ELEM_GET_WEIGHT(transition_elem);

    if (placepart != PT_TRANSITION_LINE_END) {
      warp_marking_buf[placepart] += weightpart;
    }
  }

  return true;
}

/** 
 * be_warp_try_fire - Check if a warp can fire a transition
 * @warp_marking_buf: Buffer holding the marking for each place
 * @warp_tl_pre: Pre-transition elements for the warp
 * @warp_tl_post: Post-transition elements for the warp
 * @transition_line_elems: Number of elements in the transition line
 *
 * This function checks if a warp can fire a transition by examining the
 * marking of the places involved in the transition. If all places are
 * marked, the transition can fire.
 *
 * Context: Expects warp without manually disabled lanes. 
 *          Also leaves behind a corrupted warp_marking_buf if not firable.
 *
 * Return: true if the transition can fire, false otherwise
 */
__device__ __forceinline__ 
bool be_warp_try_fire(be_place_t* warp_marking_buf, 
                      const be_transition_elem_t* warp_tl_pre,
                      const be_transition_elem_t* warp_tl_post,
                      const uint32_t transition_line_elems, 
                      const uint32_t lane) {

  bool firable = true;
  for (uint32_t i = lane; i < transition_line_elems; i += 32) {
    be_transition_elem_t transition_elem = __ldg(&warp_tl_pre[i]);
    if (transition_elem != BE_TRANSITION_LINE_END) { /* skip end marker */
      if (!warp_marking_buf[transition_elem]) {
        firable = false;
        continue;
      }
      warp_marking_buf[transition_elem] = 0; /* decrement buffer */
    }
  }

  __syncwarp();
  if (!__all_sync(__activemask(), firable)) return false;
  
  for (uint32_t i = lane; i < transition_line_elems; i += 32) {
    be_transition_elem_t transition_elem = __ldg(&warp_tl_post[i]);
    if (transition_elem != BE_TRANSITION_LINE_END) {
      if (warp_marking_buf[transition_elem]) {
        firable = false;
        continue;
      }
      warp_marking_buf[transition_elem] = BE_MARKED;
    }
  }

  __syncwarp();
  return __all_sync(__activemask(), firable);
}

/** 
 * minibe_warp_try_fire - Check if a warp can fire a transition in a minibe net
 * @warp_marking_buf: Buffer holding the marking for each place
 * @warp_tl_pre: Pre-transition elements for the warp
 * @warp_tl_post: Post-transition elements for the warp
 * @transition_line_elems: Number of elements in the transition line
 *
 * This function checks if a warp can fire a transition by examining the
 * marking of the places involved in the transition. If all places are
 * marked, the transition can fire.
 *
 * Context: Expects warp without manually disabled lanes. 
 *          Also leaves behind a corrupted warp_marking_buf if not firable.
 *
 * Return: true if the transition can fire, false otherwise
 */
__device__ __forceinline__ 
bool minibe_warp_try_fire(minibe_placechunk_t* warp_marking_buf, 
                          const minibe_transitionchunk_t* warp_tl_pre,
                          const minibe_transitionchunk_t* warp_tl_post,
                          const uint32_t transition_line_elems, 
                          const uint32_t lane) {

  bool firable = true;
  for (uint32_t i = lane; i < transition_line_elems; i += 32) {
    minibe_transitionchunk_t transition_mask = __ldg(&warp_tl_pre[i]);
    if (transition_mask != MINIBE_TRANSITION_LINE_END) { /* skip end marker */
      if ((warp_marking_buf[i] & transition_mask) != transition_mask) {
        firable = false;
        continue;
      }
      warp_marking_buf[i] ^= transition_mask;
    }
  }

  __syncwarp();
  if (!__all_sync(__activemask(), firable)) return false;
  
  for (uint32_t i = lane; i < transition_line_elems; i += 32) {
    minibe_transitionchunk_t transition_mask = __ldg(&warp_tl_post[i]);
    if (transition_mask != MINIBE_TRANSITION_LINE_END) {
      if ((warp_marking_buf[i] & transition_mask) != 0) {
        firable = false;
        continue;
      }
      warp_marking_buf[i] |= transition_mask;
    }
  }

  __syncwarp();
  return __all_sync(__activemask(), firable);
}


/** 
 * kk_warp_try_fire - Check if a warp can fire a transition
 * @warp_marking_buf: Buffer holding the marking for each place
 * @warp_tl_pre: Pre-transition elements for the warp
 * @warp_tl_post: Post-transition elements for the warp
 * @transition_line_elems: Number of elements in the transition line
 * @glob_color_degree: Global color degree of the net
 *
 * This function checks if a warp can fire a transition by examining the
 * marking of the places involved in the transition. If all places have
 * sufficient tokens for each color, the transition can fire.
 *
 * Context: Expects warp without manually disabled lanes. 
 *          Also leaves behind a corrupted warp_marking_buf if not firable.
 *
 * Return: true if the transition can fire, false otherwise
 */
__device__ __forceinline__
bool kk_warp_try_fire(kk_place_subelem_t* warp_marking_buf,
                      const kk_transition_subelem_t* warp_tl_pre,
                      const kk_transition_subelem_t* warp_tl_post,
                      const uint32_t transition_line_elems,
                      const uint32_t glob_color_degree,
                      const uint32_t lane) {

  const uint32_t stride = glob_color_degree + 1;
  const uint32_t strides_per_load = 32 / stride;
  const uint32_t elems_per_load = strides_per_load * stride;
  
  bool firable = true;
  bool hit_end = false;

  for (uint32_t base = 0; base < transition_line_elems && !hit_end;
      base += elems_per_load) {

    kk_transition_subelem_t elem = (base + lane < transition_line_elems) 
                                    ? __ldg(&warp_tl_pre[base + lane]) 
                                    : KK_TRANSITION_LINE_END;
    
    uint32_t stride_idx = lane / stride;
    uint32_t pos_in_stride = lane % stride;
    
    if (stride_idx < strides_per_load) {
      uint32_t place_ref_lane = stride_idx * stride;
      kk_transition_subelem_t pi_ref = __shfl_sync(__activemask(), elem,
                                                   place_ref_lane);
      
      if (pi_ref == KK_TRANSITION_LINE_END) hit_end = true;
      
      if (!hit_end && pos_in_stride > 0 && pos_in_stride <= glob_color_degree) {
        kk_transition_subelem_t pi_cj_weight = elem;
        uint32_t cj = pos_in_stride - 1;
        
        if (pi_cj_weight != 0) {
          kk_place_subelem_t* marking_ptr = 
            KK_PLACE_GET_SUBELEM_PTR(warp_marking_buf, pi_ref, cj,
                                     glob_color_degree);
          
          *marking_ptr -= pi_cj_weight;
          if (*marking_ptr < 0) {
            firable = false;
          }
        }
      }
    }
    
    hit_end = __any_sync(__activemask(), hit_end);
  }

  __syncwarp();
  if (!__all_sync(__activemask(), firable)) return false;


  hit_end = false;
  for (uint32_t base = 0; base < transition_line_elems && !hit_end;
       base += elems_per_load) {

    kk_transition_subelem_t elem = (base + lane < transition_line_elems) 
                                    ? __ldg(&warp_tl_post[base + lane]) 
                                    : KK_TRANSITION_LINE_END;
    
    uint32_t stride_idx = lane / stride;
    uint32_t pos_in_stride = lane % stride;
    
    if (stride_idx < strides_per_load) {
      uint32_t place_ref_lane = stride_idx * stride;
      kk_transition_subelem_t pi_ref = __shfl_sync(__activemask(), elem,
                                                   place_ref_lane);
      
      if (pi_ref == KK_TRANSITION_LINE_END) {
        hit_end = true;
      }
      
      if (!hit_end && pos_in_stride > 0 && pos_in_stride <= glob_color_degree) {
        kk_transition_subelem_t pi_cj_weight = elem;
        uint32_t cj = pos_in_stride - 1;
        
        if (pi_cj_weight != 0) {
          kk_place_subelem_t* marking_ptr = 
            KK_PLACE_GET_SUBELEM_PTR(warp_marking_buf, pi_ref, cj, 
                                     glob_color_degree);
          
          *marking_ptr += pi_cj_weight;
        }
      }
    }
    
    hit_end = __any_sync(__activemask(), hit_end);
  }

  return true;
}

/**
 * insert_initial_node_kernel - Insert the initial node into the node array
 * @dev_nodes_arr: Pointer to the device node array
 * @initial_marking: Pointer to the initial marking
 * @dev_nodewrite_locks: Pointer to the device node write locks
 * @bytes_per_node: Number of bytes per node
 * @maxnodes: Maximum number of nodes in the node array
 * @dev_f: Pointer to the frontier structure on device
 *
 * This kernel inserts the initial node into the device node array,
 * computes its hash, and updates the frontier structure accordingly.
 *
 * Return: void
 */
__global__ void insert_initial_node_kernel(void* dev_nodes_arr,
                                           void* initial_marking,
                                           uint32_t* dev_nodewrite_locks,
                                           uint32_t bytes_per_node,
                                           uint32_t maxnodes,
                                           frontier_t* dev_f) {

  const uint32_t lane = threadIdx.x & 31;
  const uint32_t uint64s_per_node = bytes_per_node / sizeof(uint64_t);
  
  uint64_t* initial_marking_64 = (uint64_t*)initial_marking;
  
  uint64_t initial_hash = warp_murmur3_hash(initial_marking_64, 
                                          uint64s_per_node, lane);
  
  initial_hash &= (maxnodes - 1);
  uint64_t* target_node_ptr = (uint64_t*)((uint8_t*)dev_nodes_arr 
                                          + initial_hash * bytes_per_node);
  for (uint32_t i = lane; i < uint64s_per_node; i += 32) {
    target_node_ptr[i] = initial_marking_64[i];
  }
  __syncwarp();
  if (lane == 0) {
    dev_nodewrite_locks[initial_hash] = LOCK_WRITTEN;
    
    dev_f->front_now[0] = (uint32_t)initial_hash;
    dev_f->head_now = 1;
  }
}

/**
 * rg_transition_kernel - Simulator kernel for reachability graphs
 * @net_type_t: Template - Type of the net (e.g., pt_net_t)
 * @use_shared: Template - Whether to use shared memory for marking buffers
 * @rg_data: Reachability graph data structure
 * @dev_created_nodes: Pointer to the atomic counter for created nodes
 * @dev_created_arcs: Pointer to the atomic counter for created arcs
 * @dev_nodewrite_locks: Pointer to the device node write locks
 * @dev_f: Pointer to the frontier structure on device
 * @dev_global_runflag: Pointer to the global run flag on device
 * @dev_global_sync_lock: Pointer to the global synchronization lock on device
 * @shared_mem_size: Size of shared memory available per block
 * @glob_pool: Pointer to the global memory pool for marking buffers
 *
 * This kernel simulates the firing of transitions in a reachability graph.
 * One thread handles one transition and works on its locality. It shares the
 * implementation for the different net types (BE, PT, KK) by using compile-time
 * polymorphism.
 * 
 * The node generation is in BFS-order, where each iteration step generates n
 * nodes in the node array, and all threads fetch the same node (i.e. marking),
 * and then try to fire all transitions, generating new nodes.
 * These nodes get "pushed back" through a queue-like counter dev_created_nodes,
 * pointing to the next free node in the node array, and the iterator, which
 * points to the visited nodes in the node array. 
 * We abuse the fact that
 * we walk through the max_nodes array, each transition at the same marking
 * one transition writes the new node, all fetch it into their array
 * -> we could have all threads have a buffer, copy the node into it,
 *    modify it, and release the node
 * -> issue: per-thread buffer must be in shared - or slow! - big buffers
 *           do not fit into shared memory
 * -> solution: all threads check activation for node in node array,
 *              for n activated threads, the node is n-times replicated,
 *              activate threads (in low-id->high-id) patch their new node
 *              and also release their new arc
 * -> new issue: mitigation of duplicate arc
 * -> solution: fall back to solution one, due to the need of having a 
 *              buffer, where we write the post-fired marking to for
 *              comparison with the full node array
 *
 * Context: This kernel is called from the host with the number of maxnodes to
 *          simulate, and it runs until all maxnodes are consumed.
 * 
 * Return: void
 */
template<typename net_type_t, bool use_shared>
static __global__ void rg_transition_kernel(rg_data_t rg_data,
                                            uint32_t* dev_created_nodes,
                                            uint32_t* dev_created_arcs,
                                            uint32_t* dev_nodewrite_locks,
                                            frontier_t* dev_f,
                                            uint32_t* dev_global_runflag,
                                            uint32_t* dev_global_sync_lock,
                                            uint32_t  shared_mem_size,
                                            void* glob_pool = nullptr) {

  extern __shared__ uint8_t shared[]; /* this is used differently so byte buf */

  /* warp in group/block does ONE transition (and some more)             */
  /* group/block does ALL transitions per node, so full and single visit */

  const uint32_t worker = threadIdx.x;
  const uint32_t lane = worker & 31;
  const uint32_t group =  blockIdx.x;
  const uint32_t warp_in_group = worker >> 5;
  const uint32_t total_warps_per_group = blockDim.x >> 5;
  const uint32_t total_groups = gridDim.x;

  /* we can not have inactive warps, this is guaranteed */
  /* due to reduction of warps for less transitions     */

#define DO_ONCE(X) { if (group == 0 && worker == 0) { X; } }
#define DO_ONCE_INWARP(X) { if (lane == 0) { X; } __syncwarp(); }
#define DO_ONCE_INGROUP(X) { if (worker == 0) { X; } __syncthreads(); }

#ifdef DEBUG
  char transition_name[64];
  char *p = transition_name;
  *p++ = 'W';
  p += int_to_str(p, worker);
  *p++ = '-';
  *p++ = '(';
  *p++ = 'G';
  p += int_to_str(p, group);
  *p++ = ')';
  *p = '\0';
#endif

  const uint32_t bytes_per_node = rg_data.bytes_per_node;
  const uint32_t maxnodes = rg_data.maxnodes;
  const uint32_t transition_line_elems = rg_data.transition_line_elems;

  /* every warp in a group must fit its own scratchpad into shared mem */
  if constexpr(use_shared){
    if (total_warps_per_group * bytes_per_node > shared_mem_size - 4) {
      DO_ONCE(printf("Error: Not enough shared memory per block, %u > %u!\n",
                     total_warps_per_group * bytes_per_node, 
                     shared_mem_size -4));
      return;
    }
  }

  uint64_t* warp_marking_buf_64;
  if constexpr (use_shared){
    warp_marking_buf_64 = (uint64_t*)((uint8_t*) shared 
                                       + warp_in_group * bytes_per_node);
  } else {
    const uint32_t total_warps_per_group = blockDim.x >> 5;
    const uint32_t global_warp_id = (blockIdx.x * total_warps_per_group) 
                                    + warp_in_group;
    warp_marking_buf_64 = (uint64_t*)((uint8_t*) glob_pool 
                                       + global_warp_id * bytes_per_node);
  }

  const uint32_t uint64s_per_node  = rg_data.bytes_per_node / sizeof(uint64_t); 
  uint32_t elements_per_node;
  if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
    elements_per_node = rg_data.bytes_per_node / sizeof(pt_place_t);
  } else if constexpr (SAME_TYPE(net_type_t, be_net_t)) {
    elements_per_node = rg_data.bytes_per_node / sizeof(be_place_t);
  } else if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
    elements_per_node = rg_data.bytes_per_node / sizeof(kk_place_subelem_t);
  } else if constexpr (SAME_TYPE(net_type_t, minibe_net_t)) {
    elements_per_node = rg_data.bytes_per_node / sizeof(minibe_placechunk_t);
  }

  /* bytes per node is padded to 128, so calc is fine */

  void* nodes_ptr = (void*) rg_data.dev_nodes_arr;

  void* dev_transitions_arr = (void*) rg_data.transition_arr;

  /* visiting idx in node arr */
  uint32_t* shared_v;
  if constexpr (use_shared) {
    shared_v = (uint32_t*)&shared[shared_mem_size - sizeof(uint32_t)];
  } else {
    shared_v = (uint32_t*)shared;
  }

  DO_ONCE_INGROUP({
    *shared_v = 0xFFFFFFFF;
  });

  uint32_t local_sense = 1;

  /************************************************************* setup done ***/

  while (true) {

    if (dev_f->head_now == 0) break;

    for (uint32_t front_idx = group; front_idx < dev_f->head_now; 
         front_idx += total_groups) {

      if (atomicAdd(dev_global_runflag, 0) == 0) break;

      DO_ONCE_INGROUP({
        *shared_v = dev_f->front_now[front_idx];
        DEBUG_PRINTF("(%s) Front idx: %u, Node idx: %u, head_now: %d, "
                     "nodes_ptr: %p, offset: %lu, address: %p\n", 
                     transition_name, 
                     front_idx, *shared_v, dev_f->head_now, nodes_ptr,
                     (uint64_t)(*shared_v * elements_per_node),
                     nodes_ptr + *shared_v * elements_per_node);
      });

      const uint64_t* c_node_u64 = (uint64_t*)((uint8_t*)rg_data.dev_nodes_arr 
                                                         + *shared_v 
                                                         * bytes_per_node);

      for (uint32_t tid = warp_in_group; tid < rg_data.transitions; 
           tid += total_warps_per_group) {

        if (atomicAdd(dev_global_runflag, 0) == 0) break;

        void* warp_tl_pre;
        void* warp_tl_post;
        if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
          warp_tl_pre = &((pt_transition_elem_t*)dev_transitions_arr)[
            tid * transition_line_elems * 2
          ];
          warp_tl_post = (((pt_transition_elem_t*)warp_tl_pre) 
                          + transition_line_elems);
        } else if constexpr (SAME_TYPE(net_type_t, be_net_t)) {
          warp_tl_pre = &((be_transition_elem_t*)dev_transitions_arr)[
            tid * transition_line_elems * 2
          ];
          warp_tl_post = (((be_transition_elem_t*)warp_tl_pre) 
                          + transition_line_elems);
        } else if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
          warp_tl_pre = &((kk_transition_subelem_t*)dev_transitions_arr)[
            tid * transition_line_elems * 2
          ];
          warp_tl_post = (((kk_transition_subelem_t*)warp_tl_pre) 
                          + transition_line_elems);
        } else if constexpr (SAME_TYPE(net_type_t, minibe_net_t)) {
          warp_tl_pre = &((minibe_transitionchunk_t*)dev_transitions_arr)[
            tid * transition_line_elems * 2
          ];
          warp_tl_post = (((minibe_transitionchunk_t*)warp_tl_pre) 
                          + transition_line_elems);
        }

        warp_aligned_copy(warp_marking_buf_64, c_node_u64, uint64s_per_node,
                          lane);

        {
          bool c;
          
          if constexpr(SAME_TYPE(net_type_t, pt_net_t)) {
            c = !pt_warp_try_fire((pt_place_t*)warp_marking_buf_64,
                                  (pt_transition_elem_t*)warp_tl_pre,
                                  (pt_transition_elem_t*)warp_tl_post,
                                  transition_line_elems,
                                  lane);
          } else if constexpr(SAME_TYPE(net_type_t, be_net_t)) {
            c = !be_warp_try_fire((be_place_t*)warp_marking_buf_64,
                                  (be_transition_elem_t*)warp_tl_pre,
                                  (be_transition_elem_t*)warp_tl_post,
                                  transition_line_elems,
                                  lane);
          } else if constexpr(SAME_TYPE(net_type_t, kk_net_t)) {
            c = !kk_warp_try_fire((kk_place_subelem_t*)warp_marking_buf_64,
                                  (kk_transition_subelem_t*)warp_tl_pre,
                                  (kk_transition_subelem_t*)warp_tl_post,
                                  transition_line_elems,
                                  rg_data.glob_color_degree,
                                  lane);
          } else if constexpr(SAME_TYPE(net_type_t, minibe_net_t)) {
            c = !minibe_warp_try_fire((minibe_placechunk_t*)warp_marking_buf_64,
                                      (minibe_transitionchunk_t*)warp_tl_pre,
                                      (minibe_transitionchunk_t*)warp_tl_post,
                                      transition_line_elems,
                                      lane);
          }

          if (c) continue; /* new warp buf + new trans/node */
        }

        uint64_t new_node_hash = warp_murmur3_hash(warp_marking_buf_64, 
                                                   uint64s_per_node, lane);

#ifdef DEBUG
        DO_ONCE_INWARP({
          DEBUG_PRINTF("(%s) TID %d: Fired transition %u at node %u, "
                       "new node hash: %lu\n", 
                       transition_name, tid, tid, *shared_v, new_node_hash);
        });
#endif                         

        /* now new hash is equal across warp, lets put our stuff in node */
        for (uint32_t probe = 0; probe < maxnodes; probe += 1) {

          const uint32_t target_node_idx = (new_node_hash + probe)
                                  &(maxnodes-1);

          uint64_t* target_node_ptr_64;
          if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
            target_node_ptr_64 = (uint64_t*)((pt_place_t*)nodes_ptr 
                                             + target_node_idx 
                                             * elements_per_node);
          } else if constexpr (SAME_TYPE(net_type_t, be_net_t)) {
            target_node_ptr_64 = (uint64_t*)((be_place_t*)nodes_ptr 
                                             + target_node_idx 
                                             * elements_per_node);
          } else if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
            target_node_ptr_64 = (uint64_t*)((kk_place_subelem_t*)nodes_ptr 
                                             + target_node_idx 
                                             * elements_per_node);
          } else if constexpr (SAME_TYPE(net_type_t, minibe_net_t)) {
            target_node_ptr_64 = (uint64_t*)((minibe_placechunk_t*)nodes_ptr 
                                             + target_node_idx 
                                             * elements_per_node);
          }

          uint32_t* target_lock = &dev_nodewrite_locks[target_node_idx];

#ifdef DEBUG
          DO_ONCE_INWARP({
            DEBUG_PRINTF("TID %d: Comparing at probe %u, lock=%u\n",
                          tid, probe, *target_lock);
          });
#endif 

          uint32_t lock_state;
          

          do {
            DO_ONCE_INWARP(lock_state = atomicCAS(target_lock, LOCK_FREE, 
                                                  LOCK_WRITING));
            lock_state = __shfl_sync(__activemask(), lock_state, 0);
          } while (lock_state == LOCK_WRITING);

          if (lock_state == LOCK_WRITTEN) {
            /* make comparison, here a written node is living, full comp... */
            bool is_equal = warp_arrays_equal(target_node_ptr_64, 
                                              warp_marking_buf_64,
                                              uint64s_per_node, lane);
            if (is_equal) {
              DO_ONCE_INWARP(arc_publish(rg_data.dev_arcs_arr, *shared_v,
                                        target_node_idx, tid, dev_created_arcs,
                                        rg_data.maxarcs));
              break; /* from probing => get next transition/node */
            } /* else continue probing */

          } 
          if (lock_state == LOCK_FREE) {
            /* real release */
            
            /************************************************ maxnode guard ***/
            uint32_t old_node_count;
            DO_ONCE_INWARP({
              old_node_count = atomicAdd(dev_created_nodes, 1);
            });
            old_node_count = __shfl_sync(__activemask(), old_node_count, 0);
            
            if (old_node_count >= rg_data.maxnodes) {
              DO_ONCE_INWARP({
                atomicSub(dev_created_nodes, 1);
                /* notify warp and exit */
                atomicExch(target_lock, LOCK_FREE);
                atomicExch(dev_global_runflag, 0);
              }); 
              break;
            }

            warp_aligned_copy(target_node_ptr_64, warp_marking_buf_64, 
                              uint64s_per_node, lane);

            DO_ONCE_INWARP({
              atomicExch(target_lock, LOCK_WRITTEN); /* now others may compare */
              /* can not overflow due to created nodes boundary check above */
              frontier_push_next(dev_f, target_node_idx);
              arc_publish(rg_data.dev_arcs_arr, *shared_v, target_node_idx, tid, 
                          dev_created_arcs, rg_data.maxarcs);
            });

            break; /* from probing => get next transition/node */

          } else if (lock_state == LOCK_WRITING) {
            while (atomicAdd(target_lock, 0) == LOCK_WRITING) __threadfence();
            /* somebody wrote to it, we waited and now proceed for compare */
          } 

        } /* end probing => get next transition/node */

        /* generated new arc, shared buf is trash */
        /* continue for next transition or new node */

      } /* walked through all transitions, meaning full node done */


    } /* walked through frontier, group has too big idx */

#ifdef DEBUG
    DO_ONCE({
        DEBUG_PRINTF("Before switch: head_now=%d, head_next=%d\n", 
                     dev_f->head_now, dev_f->head_next);
    });
#endif

    glob_syncd_frontier_switch(dev_global_sync_lock, total_groups * blockDim.x,
                               local_sense, dev_f);


#ifdef DEBUG
    DO_ONCE({
        DEBUG_PRINTF("After switch: head_now=%d, head_next=%d\n", 
                     dev_f->head_now, dev_f->head_next);
    });
#endif
  }
}

/* -------------------------------------------------------------------------- */
/* ------------------------------- main part -------------------------------- */
/* --------------------------------- host ----------------------------------- */
/* -------------------------------------------------------------------------- */


/* used for time calculation before kernel execution and during */
struct timespec start, prekernel, postkernel;

/**
 * dissect_netdata_run_rg_gen - runs the reachability graph generator kernel
 * @dev_props: pointer to the device properties structure
 * @managed_filedata_ptr: pointer to the managed file data, which means its
 *                       living in memory both accessible by device and host
 * @netbytes_size: size of the net data in bytes, which is the file-input size
 * @maxnodes: maximum number of nodes to be generated in the reachability graph
 * @nodes_out_file: file to write the nodes of the reachability graph to
 * @arcs_out_file: file to write the arcs of the reachability graph to
 *
 * This function sets up the device memory for the reachability graph, allocates
 * the fragmented transition texture, and runs the reachability graph generator
 * kernel.
 * It only operates on device memory, which means that the net file data comes
 * from managed memory, is copied to the device, processed and copied back to
 * host memory.
 * Here, we do not need to fully recopy the managed_filedata_ptr to the device,
 * we only need the textures of the transitions, and the parsed metadata.
 * The marking data is not operated on permanently, we need to allocate a
 * big node structure where the markings will be written to.
 *
 * Context: This function is called under the following assumptions:
 *            - device properties are fetched (dev_props)
 *            - the net data is ready to be processed and lives in memory that
 *              is readable from the host (and device if wanted, mode-specific)
 *            - the maximum number of nodes to be generated is parsed from CLI
 *              (maxnodes)
 *           - the output files for the nodes and arcs are specified
 * 
 * Return: 0 on success, 1 on failure
 */
static uint8_t dissect_netdata_run_rg_gen(device_prop_t* dev_props,
                                          uint8_t* managed_filedata_ptr,
                                          uint32_t maxnodes, uint32_t maxarcs,
                                          FILE* nodes_out_file,
                                          FILE* arcs_out_file) {

  if (nodes_out_file == 0 || arcs_out_file == 0) {
    printf("WARNING: an out-file must be specified in this mode!\n");
    exit(1);
  }

  /* stream setup for async memcpy from managed host mem to GPU */
  cudaStream_t compute_stream, io_stream;
  cudaStreamCreate(&compute_stream);
  cudaStreamCreate(&io_stream);

  /* create the data that is worked on by the transition threads */
  rg_data_t rg_data = rg_data_t{};
  uint32_t* metadata_ptr = (uint32_t*) managed_filedata_ptr;
  rg_data.places = metadata_ptr[0];
  rg_data.transitions = metadata_ptr[1];
  rg_data.transition_line_elems = metadata_ptr[2];
  /* magic number is last uint32_t in metadata region */
  rg_data.magic_number = metadata_ptr[31];

  uint8_t net_type = rg_data.magic_number;
  if (rg_data.magic_number != NET_TYPE_PT 
      && rg_data.magic_number != NET_TYPE_KK) net_type = NET_TYPE_BE;
  /* this is because magic_number is not available in be nets */
  if (net_type == NET_TYPE_BE 
      && metadata_ptr[30] == NET_SPECIALFIELD_MINIBE_INDICATOR) {
        net_type = NET_TYPE_MINIBE;
  }

  printf("Net input is of type: ");
  if (net_type == NET_TYPE_BE) printf(NET_NAME_BE); 
  else if (net_type == NET_TYPE_PT) printf(NET_NAME_PT); 
  else if (net_type == NET_TYPE_KK) printf(NET_NAME_KK);
  else if (net_type == NET_TYPE_MINIBE) printf(NET_NAME_MINIBE);
  printf("\n");

  if (net_type == NET_TYPE_KK) 
    rg_data.glob_color_degree = metadata_ptr[3];
  else rg_data.glob_color_degree = 1;

  const float place_size = 
  rg_data.glob_color_degree * (net_type == NET_TYPE_BE ? sizeof(be_place_t) 
                               : (net_type == NET_TYPE_PT ? sizeof(pt_place_t) 
                         : (net_type == NET_TYPE_KK ? sizeof(kk_place_subelem_t) 
                               : (net_type == NET_TYPE_MINIBE 
                                  ? sizeof(minibe_placechunk_t)/64.0f : 0))));

  if (rg_data.glob_color_degree > 31) {
    printf("Error: Color degree too high (%u > 31)!\n", 
           rg_data.glob_color_degree);
    exit(EXIT_FAILURE);
  }

  uint32_t transitions_region_size = 0;
  if (net_type == NET_TYPE_BE) {
    transitions_region_size = rg_data.transitions * 
                              (2 * rg_data.transition_line_elems * 
                               sizeof(be_transition_elem_t));
  } else if (net_type == NET_TYPE_PT) {
    transitions_region_size = rg_data.transitions * 
                              (2 * rg_data.transition_line_elems * 
                               sizeof(pt_transition_elem_t));
  } else if (net_type == NET_TYPE_KK) {
    transitions_region_size = rg_data.transitions * 
                              (2 * rg_data.transition_line_elems * 
                               sizeof(kk_transition_subelem_t));
  } else if (net_type == NET_TYPE_MINIBE) {
    transitions_region_size = rg_data.transitions * 
                              (2 * rg_data.transition_line_elems * 
                               sizeof(minibe_transitionchunk_t));
  } else {
    printf("Error: Unknown net type in reachability graph generation!\n");
    exit(EXIT_FAILURE);
  }

  printf("Places: %u, Transitions: %u, Indices in transition line: %u\n",
    rg_data.places, rg_data.transitions, rg_data.transition_line_elems);

  /* the following pointer points to where the marking array is in the alloca-*/
  /* ted data, which is the start of the data chunk + metadata                */
  void* managed_marking_ptr = (void*) (managed_filedata_ptr + METADATA_PADDING);
  
  const uint32_t marking_area_bytes = (uint32_t) ROUNDF_UP(rg_data.places 
                                                           * place_size,
                                                           PLACE_PADDING);
  const uint32_t pre_transition_area_bytes = 
    ROUND_UP(METADATA_PADDING + marking_area_bytes, TRANSITION_AREA_PADDING); 
  void* managed_transitions_arr = (void*) (managed_filedata_ptr 
                                          + pre_transition_area_bytes);

  /* move it to dev */
  cudaMallocAsync((void**)&rg_data.transition_arr, 
                  transitions_region_size, io_stream);
  if (rg_data.transition_arr == nullptr) {
    printf("Error: cudaMalloc failed to allocate device transition array.\n");
    exit(EXIT_FAILURE);
  }
  cudaMemcpyAsync(rg_data.transition_arr, managed_transitions_arr,
                  transitions_region_size, cudaMemcpyHostToDevice, io_stream);

  if ((uintptr_t) rg_data.transition_arr 
      % alignof(uint64_t) != 0) {
    printf("Error: transition area not aligned perfectly.!\n");
    exit(EXIT_FAILURE);
  }

  /***************************************************** rg specific allocs ***/

  /* unused due to specific shared memory specifications */
  /* for correctness */
  cudaFuncSetCacheConfig(rg_transition_kernel<pt_net_t, true>,
                         cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(rg_transition_kernel<be_net_t, true>, 
                         cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(rg_transition_kernel<kk_net_t, true>, 
                         cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(rg_transition_kernel<minibe_net_t, false>,
                         cudaFuncCachePreferL1);

  rg_data.bytes_per_node = marking_area_bytes;
  rg_data.maxnodes = next_power_of_two((uint32_t)(maxnodes * 1.5f));
  cudaMallocAsync((void**)&rg_data.dev_nodes_arr, 
                  rg_data.bytes_per_node * rg_data.maxnodes, io_stream);

  if (rg_data.dev_nodes_arr == nullptr) {
    printf("Error: cudaMalloc failed to allocate device nodes array.\n");
    exit(EXIT_FAILURE);
  }
  rg_data.maxarcs = maxarcs;
  cudaMallocAsync((void**)&rg_data.dev_arcs_arr, 
                  rg_data.maxarcs * sizeof(rg_arc_t), io_stream);

  if (rg_data.dev_arcs_arr == nullptr) {
    printf("Error: cudaMalloc failed to allocate device arcs array.\n");
    exit(EXIT_FAILURE);
  }
  cudaMemsetAsync(rg_data.dev_nodes_arr, 0xff, 
                  rg_data.bytes_per_node * rg_data.maxnodes, io_stream);
  cudaMemsetAsync(rg_data.dev_arcs_arr, 0xff,
                  rg_data.maxarcs * sizeof(rg_arc_t), io_stream);

  const uint32_t one = 1;
  /* global tracker for the number of created nodes for dev_nodes_arr, */
  /* which holds the index at which the next node is to be created     */
  uint32_t* dev_created_nodes;
  cudaMallocAsync(&dev_created_nodes, sizeof(uint32_t), io_stream);
  if (dev_created_nodes == nullptr) {
    printf("Error: cudaMalloc failed to allocate device nodes counter.\n");
    exit(EXIT_FAILURE);
  }
  cudaMemcpyAsync(dev_created_nodes, &one, sizeof(uint32_t),
                  cudaMemcpyHostToDevice, io_stream);

  uint32_t* dev_created_arcs;
  cudaMallocAsync(&dev_created_arcs, sizeof(uint32_t), io_stream);
  if (dev_created_arcs == nullptr) {
    printf("Error: cudaMalloc failed to allocate device arcs counter.\n");
    exit(EXIT_FAILURE);
  }
  cudaMemsetAsync(dev_created_arcs, 0, sizeof(uint32_t), io_stream);

  uint32_t* dev_nodewrite_locks;
  cudaMallocAsync(&dev_nodewrite_locks,
                  sizeof(uint32_t) * rg_data.maxnodes, io_stream);

  if (dev_nodewrite_locks == nullptr) {
    printf("Error: cudaMalloc failed to allocate device node write lock.\n");
    exit(EXIT_FAILURE);
  }
  cudaMemsetAsync(dev_nodewrite_locks, 0, 
                  rg_data.maxnodes * sizeof(uint32_t), io_stream);

  uint32_t* dev_global_runflag;
  cudaMallocAsync(&dev_global_runflag, sizeof(uint32_t), io_stream);
  if (dev_global_runflag == nullptr) {
    printf("Error: cudaMalloc failed to allocate device global runflag.\n");
    exit(EXIT_FAILURE);
  }
  cudaMemcpyAsync(dev_global_runflag, &one, sizeof(uint32_t),
                  cudaMemcpyHostToDevice, io_stream);

  uint32_t* dev_global_sync_lock;
  cudaMallocAsync(&dev_global_sync_lock, 2 * sizeof(uint32_t), io_stream);
  if (dev_global_sync_lock == nullptr) {
    printf("Error: cudaMalloc failed to allocate device global syncflag.\n");
    exit(EXIT_FAILURE);
  }
  cudaMemsetAsync(dev_global_sync_lock, 0, 2 * sizeof(uint32_t), io_stream);

  frontier_t* dev_f = frontier_devcreate(rg_data.maxnodes);

  cudaStreamSynchronize(io_stream);

  insert_initial_node_kernel<<<1, 32, 0, io_stream>>>(rg_data.dev_nodes_arr,
                                                      managed_marking_ptr,
                                                      dev_nodewrite_locks,
                                                      rg_data.bytes_per_node,
                                                      rg_data.maxnodes,
                                                      dev_f);

  cudaStreamSynchronize(io_stream);
  printf("Initial node inserted.\n");
  
  /********************************************************* kernel prepare ***/

  int32_t min_grid_sz;
  int32_t opt_block_size;
  uint32_t shared_mem_size = dev_props->max_shared_per_block;

  cudaError_t err;

  err = cudaOccupancyMaxPotentialBlockSize(&min_grid_sz, &opt_block_size, 
                                      net_type == NET_TYPE_BE ?
                                          rg_transition_kernel<be_net_t, true>:
                                      net_type == NET_TYPE_PT ?
                                          rg_transition_kernel<pt_net_t, true>:
                                      net_type == NET_TYPE_KK ?
                                          rg_transition_kernel<kk_net_t, true>:
                                      net_type == NET_TYPE_MINIBE ?
                                       rg_transition_kernel<minibe_net_t, true>:
                                          nullptr);

  if (err != cudaSuccess) {
      fprintf(stderr, "Occupancy calculation failed: %s\n",
              cudaGetErrorString(err));
      exit(1);
  }

  int32_t blocks_per_sm; /* min block count for full gpu occupation */
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm,
    net_type == NET_TYPE_BE ? rg_transition_kernel<be_net_t, true> :
    net_type == NET_TYPE_PT ? rg_transition_kernel<pt_net_t, true> :
    net_type == NET_TYPE_KK ? rg_transition_kernel<kk_net_t, true> :
    net_type == NET_TYPE_MINIBE ? rg_transition_kernel<minibe_net_t, true> :
    nullptr,
    opt_block_size,
    shared_mem_size
  );

  if (err != cudaSuccess) {
      fprintf(stderr, "Occupancy calculation failed: %s\n",
              cudaGetErrorString(err));
      exit(1);
  }

  bool use_glob_pool = false;
  { /* constrained block adjust */
    int32_t max_resident_blocks = blocks_per_sm * dev_props->sms;
    
    if (min_grid_sz > max_resident_blocks) {
      printf("Clamping grid size from %d to %d for launch safety.\n", 
              min_grid_sz, max_resident_blocks);
      min_grid_sz = max_resident_blocks;
    }

    if (opt_block_size % 32 != 0) {
      fprintf(stderr, "Error: optimal block size is not warp-size "
                      "multiple (%d)!\n", opt_block_size);
      exit(1);
    }

    const uint32_t warps_in_block = opt_block_size >> 5;
    if (rg_data.transitions < warps_in_block) {
      printf("Reducing block size %d for transitions %d\n", opt_block_size,
            rg_data.transitions);
      opt_block_size = rg_data.transitions * 32;
    }

    const uint32_t total_warps_per_group = opt_block_size >> 5;
    if (total_warps_per_group * rg_data.bytes_per_node > shared_mem_size - 4) {
      
      shared_mem_size = sizeof(uint32_t); 
      /* do not use shared, but rather glob mem */
      printf("Launch was <<<%d, %d>>>, but not enough shared mem, recalc...\n",
             min_grid_sz, opt_block_size);

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        net_type == NET_TYPE_BE ? rg_transition_kernel<be_net_t, false> :
        net_type == NET_TYPE_PT ? rg_transition_kernel<pt_net_t, false> :
        net_type == NET_TYPE_KK ? rg_transition_kernel<kk_net_t, false> :
        net_type == NET_TYPE_MINIBE ? rg_transition_kernel<minibe_net_t, false>:
        nullptr,
        opt_block_size,
        shared_mem_size
      );
      
      if (err != cudaSuccess) {
          fprintf(stderr, "Occupancy calculation failed: %s\n",
                  cudaGetErrorString(err));
          exit(1);
      }

      /* max_resident_blocks does not need to be 1 per sm */
      max_resident_blocks = blocks_per_sm * dev_props->sms;
      use_glob_pool = true;
    }
  }

  printf("Optimal <<<%d, %d>>> launch for n*%d workers\n", min_grid_sz, 
         opt_block_size, rg_data.transitions);
  printf("Total threads: %d (waiting on start: %d)\n",
          min_grid_sz * opt_block_size,
          min_grid_sz * opt_block_size - rg_data.transitions);
  printf("Shared memory per block: %u bytes\n", shared_mem_size);

  void* glob_pool = nullptr;
  if (use_glob_pool) {
    const uint32_t warps_in_block = opt_block_size >> 5;

    const uint32_t total_warps_in_grid = min_grid_sz * warps_in_block;
    const uint32_t glob_pool_size = total_warps_in_grid   
                                    * rg_data.bytes_per_node;

    cudaMallocAsync(&glob_pool, glob_pool_size, compute_stream);
    if (glob_pool == nullptr) {
      printf("Error: cudaMalloc failed to allocate global pool memory.\n");
      exit(EXIT_FAILURE);
    }
    printf("Using global pool of size %u bytes for shared mem emulation.\n",
           glob_pool_size);
  }

  /* since we use maximum amount of threads concurrently available   */
  /*  we do not use opt block size, but rather max through dev_props */

  /* synchronize streams before kernel launch to ensure data is ready */
  cudaStreamSynchronize(io_stream);

  /************************************************************* main phase ***/
  clock_gettime(CLOCK_TYPE, &prekernel);
  double setup_time = timespec_diff_ms(&start, &prekernel);
  printf("Setup time: %.9f ms\n", setup_time);

  /************************************************************ kernel exec ***/

  void *kernel_args[] = {
    &rg_data,
    &dev_created_nodes,
    &dev_created_arcs,
    &dev_nodewrite_locks,
    &dev_f,
    &dev_global_runflag,
    &dev_global_sync_lock,
    &shared_mem_size,
    &glob_pool
  };

  if (net_type == NET_TYPE_BE) {
    if (use_glob_pool) {
      cudaLaunchKernel((void*)rg_transition_kernel<be_net_t, false>,
                       dim3(min_grid_sz), dim3(opt_block_size),
                       kernel_args, shared_mem_size, compute_stream);
    } else {
      cudaLaunchKernel((void*)rg_transition_kernel<be_net_t, true>,
                       dim3(min_grid_sz), dim3(opt_block_size),
                       kernel_args, shared_mem_size, compute_stream);
    }
    
  } else if (net_type == NET_TYPE_PT) {
    if (use_glob_pool) {
      cudaLaunchKernel((void*)rg_transition_kernel<pt_net_t, false>,
                       dim3(min_grid_sz), dim3(opt_block_size),
                       kernel_args, shared_mem_size, compute_stream);
    } else {
      cudaLaunchKernel((void*)rg_transition_kernel<pt_net_t, true>,
                       dim3(min_grid_sz), dim3(opt_block_size),
                       kernel_args, shared_mem_size, compute_stream);
    }
  } else if (net_type == NET_TYPE_KK) {
    if (use_glob_pool) {
      cudaLaunchKernel((void*)rg_transition_kernel<kk_net_t, false>,
                       dim3(min_grid_sz), dim3(opt_block_size),
                       kernel_args, shared_mem_size, compute_stream);
    } else {
      cudaLaunchKernel((void*)rg_transition_kernel<kk_net_t, true>,
                       dim3(min_grid_sz), dim3(opt_block_size),
                       kernel_args, shared_mem_size, compute_stream);
    }
  } else if (net_type == NET_TYPE_MINIBE) {
    if (use_glob_pool) {
      cudaLaunchKernel((void*)rg_transition_kernel<minibe_net_t, false>,
                       dim3(min_grid_sz), dim3(opt_block_size),
                       kernel_args, shared_mem_size, compute_stream);
    } else {
      cudaLaunchKernel((void*)rg_transition_kernel<minibe_net_t, true>,
                       dim3(min_grid_sz), dim3(opt_block_size),
                       kernel_args, shared_mem_size, compute_stream);
    }
  } else {
    printf("Error: Unknown net type in reachability graph generation!\n");
    exit(EXIT_FAILURE);
  }

  cudaStreamSynchronize(compute_stream);
  CUDA_ERR_ASSERT();

  clock_gettime(CLOCK_TYPE, &postkernel);
  double kernel_time = timespec_diff_ms(&prekernel, &postkernel);
  printf("Kernel execution time: %.9f ms\n", kernel_time);

  /********************************************************** cleanup phase ***/

  SAFE_CUDAFREE(glob_pool);
  SAFE_CUDAFREE(dev_global_sync_lock);
  SAFE_CUDAFREE(dev_global_runflag);
  SAFE_CUDAFREE(dev_created_nodes);
  SAFE_CUDAFREE(dev_created_arcs);
  SAFE_CUDAFREE(dev_nodewrite_locks);
  frontier_devdestroy(dev_f);

  void* copied_back_nodes_arr;
  void* copied_back_arcs_arr;

  cudaHostAlloc(&copied_back_nodes_arr, 
                rg_data.bytes_per_node*rg_data.maxnodes, cudaHostAllocDefault);
  cudaHostAlloc(&copied_back_arcs_arr, rg_data.maxarcs * sizeof(rg_arc_t), 
                cudaHostAllocDefault);

  cudaMemcpyAsync(copied_back_nodes_arr, rg_data.dev_nodes_arr, 
                  rg_data.bytes_per_node * rg_data.maxnodes,
                  cudaMemcpyDeviceToHost, io_stream);
  cudaMemcpyAsync(copied_back_arcs_arr, rg_data.dev_arcs_arr,
                  rg_data.maxarcs * sizeof(rg_arc_t), cudaMemcpyDeviceToHost,
                  io_stream);

  cudaStreamSynchronize(io_stream); /* wait for copy completion */

  uint32_t nodes_written = fwrite(copied_back_nodes_arr, 
                                  rg_data.bytes_per_node, 
                                  rg_data.maxnodes, nodes_out_file);

  uint32_t arcs_written = fwrite(copied_back_arcs_arr,
                                 sizeof(rg_arc_t), rg_data.maxarcs, 
                                 arcs_out_file);

  SAFE_CUDAFREE(rg_data.dev_nodes_arr);
  SAFE_CUDAFREE(rg_data.dev_arcs_arr);

  cudaFreeHost(copied_back_nodes_arr);
  cudaFreeHost(copied_back_arcs_arr);

  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(io_stream);
  return 0;
}

/**
 * main - Entry point for the KK/PT/BE-Net-Simulation Core via CUDA
 * @argc: Number of command-line arguments
 * @argv: Array of command-line arguments
 *
 * This function initializes the simulation environment, parses input arguments,
 * and launches the CUDA kernel to simulate petri nets (BE, PT, KK). It supports
 * reading input from files or standard input and writes the simulation results
 * to a file or pipe.
 *
 * Supported command-line arguments:
 *   ./executable <file-in> <steps/0> <file/pipe-out> [--remote/-r 
 *    <recv-port>, <send-to-addr:port>]
 *
 *   cat file_in | ./executable <steps/0> <file/pipe-out> [--remote/-r 
 *    <recv-port>, <send-to-addr:port>]
 *
 *   ./executable --info/-i/--device/-d
 *
 * Context: This function is the entry point for the program and handles
 *          initialization, kernel execution, and cleanup.
 *
 * Return: 0 on success, non-zero on failure.
 */
int main(int argc, char* argv[]) {
  clock_gettime(CLOCK_TYPE, &start);

  /************************************************************* init phase ***/

  printf("KK/PT/BE-Net-Reachability-Graph-Generation Core via CUDA\n");

  if (argc == 1  || (strcmp(argv[1], "--help") == 0 
                 || strcmp(argv[1], "-h") == 0)) {
    print_usage(argv[0]);
    return 1;
  }

  device_prop_t dev_props = get_device_properties();

  if (argc == 1 || strcmp(argv[argc-1], "--info") == 0 
                || strcmp(argv[argc-1], "-i") == 0) {
    print_info(&dev_props);
    return 0;
  }

  if (strcmp(argv[argc-1], "--device") == 0 
      || strcmp(argv[argc-1], "-d") == 0) {
    print_dev_info(&dev_props);
    return 0;
  }

  if (argc != 4 && argc != 5) {
    fprintf(stderr, "Error: Invalid number of arguments.\n");
    print_usage(argv[0]);
    return 1;
  }

  const uint32_t maxnodes = atoi(argv[2]);
  printf("Max nodes set to: %d\n", maxnodes);

  const uint32_t maxarcs = atoi(argv[3]);
  printf("Max arcs set to: %d\n", maxarcs);

  uint8_t* host_managed_data; /* bytes */
  uint32_t data_bytes;
  char* fileout;

  if (argc == 4) {
    data_bytes = fl_stdin_managed_malloc(&host_managed_data);
    fileout = argv[1];
  } else {
    data_bytes = fl_file_managed_malloc(argv[1], &host_managed_data);
    fileout = argv[4];
  }

  if (data_bytes == 0) {
    fprintf(stderr, "Error: Could not read input data.\n");
    return 1;
  }

  char nodes_out_file[256];
  char arcs_out_file[256];
  snprintf(nodes_out_file, sizeof(nodes_out_file), "%s-nodes", fileout);
  snprintf(arcs_out_file, sizeof(arcs_out_file), "%s-arcs", fileout);
  FILE *nodes_out = fopen(nodes_out_file, "wb");
  if (!nodes_out) {
    fprintf(stderr, "Error: Could not open output file '%s'.\n",nodes_out_file);
    return 1;
  }
  FILE *arcs_out = fopen(arcs_out_file, "wb");
  if (!arcs_out) {
    fprintf(stderr, "Error: Could not open output file '%s'.\n", arcs_out_file);
    fclose(nodes_out);
    return 1;
  }

  dissect_netdata_run_rg_gen(
    &dev_props, host_managed_data, maxnodes, maxarcs, nodes_out, arcs_out
  );

  ASSERT(fclose(nodes_out) == 0);
  ASSERT(fclose(arcs_out) == 0);

  SAFE_CUDAFREE(host_managed_data);
  return 0;
}
