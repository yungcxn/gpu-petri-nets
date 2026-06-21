/* author: Can Nayci */

/* COMPILATION FLAGS:            */
/*                               */
/* - DEBUG      (for prints)     */
/* - JAVA_SUPPORT                */

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <device_launch_parameters.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <zmq.hpp>
#include <atomic> /* TODO ! */
#include <thread>

#if defined(JAVA_SUPPORT)
#if !defined(JNI_HEADER_NAME)
#error "JNI_HEADER_NAME must be defined for Java support"
#endif

#define COMPILED_AS_LIBRARY
#include <jni.h>
/* the inclusion of the header class for the Java-Class which executes */
/*  the methods in this class are passed directly to the compiler !    */
/* the imported file name is the class, e.g. com_example_GPUConnector  */
/*  which is available under JNI_HEADER_NAME                           */
#endif

#include "headers/common.cuh"
#include "headers/fileloaderv2.cuh"

#pragma PREPROCESSOR_MARKER_BEGIN

/* COMPILATION FLAGS:            */
/*                               */
/* - DEBUG      (for prints)     */
/* - JAVA_SUPPORT                */

#define MIN_BACKOFF 32
#define MAX_BACKOFF 1024

/* if on linux */
#if defined(__linux__)
  #define CLOCK_TYPE CLOCK_MONOTONIC_RAW /* used for exact time measurements */
#else  
  #define CLOCK_TYPE CLOCK_MONOTONIC
#endif

/******************************************** simulation mode specification ***/

#define NET_NAME_BE  "BE-Net"
#define NET_NAME_SBE "BE-Net with Synchronous Transitions"
#define NET_NAME_PT  "PT-Net"
#define NET_NAME_SPT "PT-Net with Synchronous Transitions"
#define NET_NAME_KK  "KK-Net"
#define NET_NAME_SKK "KK-Net with Synchronous Transitions"

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
#define KK_TRANSITION_LINE_END 0xFFFFFFFF

#define TRANSITION_PADDING      128 /* bytes */
#define PLACE_PADDING           128
#define METADATA_PADDING        128
#define TRANSITION_AREA_PADDING 512 /* this is important for texture creation */

#define METADATA_FIELD_UNSET 0xFFFFFFFF
#define MAGIC_NUMBER_OFFSET  31

#define SIMULATION_WAIT_NANOS 1000000 /* 1ms */
#define ZMQ_ENDFETCHTIME      2000000 /* 2s */
#define NO_REMOTE_ADDR        "-"

#define ZMQ_TRANSITION_MSG_NONE     0xFFFFFFFF
#define ZMQ_TRANSITION_MSG_FIRE     0xFFFFFFFE
#define ZMQ_TRANSITION_MSG_ROLLBACK 0xFFFFFFFD
#define ZMQ_TRANSITION_DOWNLINK_PATCH_FIRE \
  0b10000000000000000000000000000000
#define ZMQ_TRANSITION_DOWNLINK_PATCH_ROLLBACK \
  0b01000000000000000000000000000000
#define ZMQ_TRANSITION_DOWNLINK_PATCH_MASK \
  0b11000000000000000000000000000000

#define RUN_COND_CTRLFLAG_IDENTIFIER 0

/* for magic number of file format */
enum _net_type_id {
  NET_TYPE_BE =  0,
  NET_TYPE_PT =  1,
  NET_TYPE_KK =  2,
  NET_TYPE_SPT = 3,
  NET_TYPE_SKK = 4,
  NET_TYPE_SBE = 5
};

enum _control_flow {
  SIMULATION_WAIT = 0,
  SIMULATION_RUN = 1,
  SIMULATION_TERMINATE = 2
};

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
typedef uint32_t kk_place_subelem_t;
typedef uint32_t kk_transition_subelem_t;

typedef std::thread           thread_t;

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

typedef union {
  struct { /* for remote places */
    /* first (and last) n places, where zmq host workers produce (consume).     */
    /* tokens to (from)                                                         */
    uint32_t zmq_in_edge_places;
    uint32_t zmq_out_edge_places;
    /* the first place of the corresponding edge gets this id, so places        */
    /* belonging globally together have correct token transfers                 */
    /* these ids will be used in the forwarded_token_t below                    */
    uint32_t zmq_in_edge_startid;
    uint32_t zmq_out_edge_startid;
  };
  struct { /* for synchronous nets with remote transitions */
    /* the last uplinks + downlinks transitions are uplinks, then downlinks */
    uint32_t zmq_uplinks;
    uint32_t zmq_downlinks;
    /* global id for tracking */
    uint32_t zmq_uplink_startid;
    uint32_t zmq_downlink_startid;
    /* to this id the first downlink transitions are mapped */
    uint32_t zmq_target_uplink_startid;
  };
} zmq_metadata_t;

template<typename net_type_t>
struct _net_data_t {
  uint32_t places;
  uint32_t transitions;
  uint32_t transition_line_elems;
  uint32_t glob_color_degree;

  zmq_metadata_t zmq_metadata; /* one 128 bit line */

  uint32_t magic_number;
  net_type_t net;
};
template<typename net_type_t>
using net_data_t = _net_data_t<net_type_t>;

typedef struct __attribute__((packed)) {
  uint32_t place_id;
  int32_t  val;
} forwarded_token_t;

typedef struct __attribute__((packed)) {
  uint32_t global_downlink_id;
  union { /* some messages between uplink and downlink do not need uplink  */
    uint32_t global_uplink_id; /* but rather a msg code (fire or rollback) */
    uint32_t msg;
  };
} uplink_downlink_msg_t;

enum _sync_transition_type {
  SYNC_TRANSITION_LOCAL = 0,
  SYNC_TRANSITION_UPLINK = 1,
  SYNC_TRANSITION_DOWNLINK = 2
};

/************************************************************* helper funcs ***/

#if !defined(COMPILED_AS_LIBRARY)
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

void print_usage(const char* progname) {
  printf(
    "Usage:\n"
    "  %s <steps/0> <file-in> <file/pipe-out> "
    "[--remote/-r <recv-port> <send-to-addr:port>]\n"
    "  cat file_in | %s <steps/0> <file/pipe-out> "
    "[--remote/-r <recv-port> <send-to-addr:port>]\n"
    "  %s --info/-i               # Print device info\n"
    "  %s --device/-d             # Print detailed device info\n"
    "  %s --help/-h               # Print this help message\n\n"
    "Arguments:\n"
    "  steps/0          Number of simulation steps (0 for interactive mode)\n"
    "  file-in          Input file or stdin pipe\n"
    "  file/pipe-out    Output file or pipe\n"
    "  --remote / -r    Enable remote mode with receive port and send address\n"
    "                   the first string after this flag is the PORT"
    " to listen for tokens\n"
    "                   the second string after this flag is the ADDRESS:PORT"
    " to send tokens to\n"
    "                   if one of the args is " NO_REMOTE_ADDR 
    " it is ignored\n"
    "\nExamples:\n"
    "  %s 100 input.cpt out\n"
    "  %s 0 input.cbe out\n"
    "  %s 0 input.ckk out --remote 9999 localhost:9998\n",
    progname, progname, progname, progname,
    progname, progname, progname, progname
  );
}
#endif

template<typename T>
static net_data_t<T> to_net_data_T(net_data_t<void_net_t> d) {
  net_data_t<T> result;
  result.places = d.places;
  result.transitions = d.transitions;
  result.transition_line_elems = d.transition_line_elems;
  result.glob_color_degree = d.glob_color_degree;
  result.magic_number = d.magic_number;
  result.zmq_metadata = d.zmq_metadata;
  result.net.dev_marking_arr = 
    reinterpret_cast<decltype(result.net.dev_marking_arr)>(
      d.net.dev_marking_arr
    );
  result.net.dev_transitions_arr = 
    reinterpret_cast<decltype(result.net.dev_transitions_arr)>(
      d.net.dev_transitions_arr
    );
  
  return result;
}


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
                                        *dev_props.max_threads_per_sm;
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

__forceinline__ static void zmq_bind_listen_port(
  zmq::socket_t& sock,
  const char* port
) {
  const char* prefix = "tcp://*:";
  char buf[50];
  strcpy(buf, prefix);
  strcat(buf, port);
  sock.bind(buf);
  usleep(1000);
}


static __forceinline__ void zmq_connect_addr_port(
  zmq::socket_t& sock, 
  const char* ipport
){
  const char* prefix = "tcp://";
  char buf[50];
  strcpy(buf, prefix);
  strcat(buf, ipport);
  sock.connect(buf);
  usleep(1000);
}


__forceinline__ static void zmq_bind_from_pair1(zmq::socket_t& sock,
                                                const char* port_comma_port) {
  const char* prefix = "tcp://*:";
  char buf[64];
  strncpy(buf, port_comma_port, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  char* port1 = strtok(buf, ",");
  if (!port1) {
    fprintf(stderr, "zmq_bind_from_pair1: invalid port string '%s'\n",
            port_comma_port);
    return;
  }
  char addr[64];
  snprintf(addr, sizeof(addr), "%s%s", prefix, port1);
  sock.bind(addr);
  usleep(1000);
}


__forceinline__ static void zmq_bind_from_pair2(zmq::socket_t& sock,
                                                const char* port_comma_port) {
  const char* prefix = "tcp://*:";
  char buf[64];
  strncpy(buf, port_comma_port, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  strtok(buf, ",");
  char* port2 = strtok(nullptr, ",");
  if (!port2) {
    fprintf(stderr, "zmq_bind_from_pair2: invalid port string '%s'\n", 
            port_comma_port);
    return;
  }
  char addr[64];
  snprintf(addr, sizeof(addr), "%s%s", prefix, port2);
  sock.bind(addr);
  usleep(1000);
}


__forceinline__ static void zmq_connect_from_pair1(
  zmq::socket_t& sock,
  const char* ip_colon_port_comma_port
) {
  const char* prefix = "tcp://";
  char buf[64];
  strncpy(buf, ip_colon_port_comma_port, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  char* ip = strtok(buf, ":");
  char* ports = strtok(nullptr, ":");
  if (!ip || !ports) {
    fprintf(stderr, "zmq_connect_from_pair1: invalid address string '%s'\n",
            ip_colon_port_comma_port);
    return;
  }
  char* port1 = strtok(ports, ",");
  if (!port1) {
    fprintf(stderr, "zmq_connect_from_pair1: invalid port string '%s'\n", 
            ip_colon_port_comma_port);
    return;
  }
  char addr[64];
  snprintf(addr, sizeof(addr), "%s%s:%s", prefix, ip, port1);
  sock.connect(addr);
  usleep(1000);
}


__forceinline__ static void zmq_connect_from_pair2(
  zmq::socket_t& sock,
  const char* ip_colon_port_comma_port
) {
  const char* prefix = "tcp://";
  char buf[64];
  strncpy(buf, ip_colon_port_comma_port, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  char* ip = strtok(buf, ":");
  char* ports = strtok(nullptr, ":");
  if (!ip || !ports) {
    fprintf(stderr, "zmq_connect_from_pair2: invalid address string '%s'\n",
            ip_colon_port_comma_port);
    return;
  }
  strtok(ports, ",");
  char* port2 = strtok(nullptr, ",");
  if (!port2) {
    fprintf(stderr, "zmq_connect_from_pair2: invalid port string '%s'\n",
            ip_colon_port_comma_port);
    return;
  }
  char addr[64];
  snprintf(addr, sizeof(addr), "%s%s:%s", prefix, ip, port2);
  sock.connect(addr);
  usleep(1000);
}


/* locks are always uint32 and not smaller since atomics only for >32bit */
static __device__ __forceinline__ bool try_lock(uint32_t* lock_ptr) {
  return atomicCAS(lock_ptr, 0, 1) == 0;
}

static __device__ __forceinline__ void release_lock(uint32_t* lock_ptr) {
  /* mustnt be atomic, we use it to release maybe-cached change to glob mem */
  atomicExch(lock_ptr, 0);
}


/* -------------------------------------------------------------------------- */
/* ------------------------------- main part -------------------------------- */
/* -------------------------------- device ---------------------------------- */
/* -------------------------------------------------------------------------- */

/**
 * kk_check_places - checks if a transition in a KK-net is activated
 * @tex_transitions: texture object for (maybe fragmented) transition data
 * @in_element0: first element of the preset line of the transition
 * @marking_array: pointer to the marking array
 * @glob_color_degree: the global color degree of the net (max cd size)
 * @transition_line_elems: number of elements one transition line (pre/postset)
 * @transition_name: name of the transition (for debugging)
 *
 * This function checks if a transition can be activated by checking the
 * markings in the marking array. It atomically tries to reserve tokens from
 * the marking array for each place's corresponding token type. If all tokens
 * got reserved, the function returns true, indicating that the transition
 * is activated, otherwise it rolls back the reserved tokens and returns false.
 * 
 * Context: This function is lockless and is to be recalled by the simulator
 * 
 * Return: true if the transition is activated, false otherwise.
 */
static __device__ __forceinline__ bool kk_check_places(
  cudaTextureObject_t tex_transitions,
  const uint32_t      in_element0,
  kk_place_subelem_t* marking_array,
  uint32_t            glob_color_degree,
  uint32_t            transition_line_elems
#if defined(DEBUG)
  , char* transition_name
#endif
) {
  const uint32_t stride = glob_color_degree + 1;
  bool success = true;
  uint32_t i_fail = 0;
  uint32_t cj_fail = 0;

  for (uint32_t i = 0; i < transition_line_elems; i += stride) {
    const uint32_t base_index = i + in_element0;
    const kk_transition_subelem_t pi_ref = 
      tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index);

    if (pi_ref == KK_TRANSITION_LINE_END) return true;

    for (uint32_t cj = 0; cj < glob_color_degree; cj++) {
      const kk_transition_subelem_t pi_cj_weight = 
        tex1Dfetch<kk_transition_subelem_t>(tex_transitions, 
                                            base_index + cj + 1);
      if (pi_cj_weight == 0) continue;
      
      kk_place_subelem_t* const marking_ptr = 
        KK_PLACE_GET_SUBELEM_PTR(marking_array, pi_ref, cj, glob_color_degree);
      
      kk_place_subelem_t current_marking = atomicAdd(marking_ptr, 0);
      bool acquired = false;

      while (current_marking >= pi_cj_weight) {
        const kk_place_subelem_t previous_marking = 
          atomicCAS(marking_ptr, current_marking, 
                    current_marking - pi_cj_weight);
        if (previous_marking == current_marking) {
          acquired = true;
          break;
        }
        current_marking = previous_marking;
      }
      
      if (!acquired) {
        success = false;
        i_fail = i;
        cj_fail = cj;
        break;
      }
    }
    if (!success) break;
  }

  if (success) return true;
  
  for (uint32_t i = 0; i < i_fail; i += stride) {
    const uint32_t base_index = i + in_element0;
    const kk_transition_subelem_t pi_ref = 
      tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index);
    if (pi_ref == KK_TRANSITION_LINE_END) break;
    for (uint32_t cj = 0; cj < glob_color_degree; cj++) {
      const kk_transition_subelem_t pi_cj_weight = 
        tex1Dfetch<kk_transition_subelem_t>(tex_transitions, 
                                            base_index + cj + 1);
      if (pi_cj_weight == 0) continue;
      kk_place_subelem_t* const marking_ptr = 
        KK_PLACE_GET_SUBELEM_PTR(marking_array, pi_ref, cj, glob_color_degree);
      atomicAdd(marking_ptr, pi_cj_weight);
    }
  }

  const uint32_t fail_base_index = i_fail + in_element0;
  const kk_transition_subelem_t fail_pi_ref = 
    tex1Dfetch<kk_transition_subelem_t>(tex_transitions, fail_base_index);
  if (fail_pi_ref != KK_TRANSITION_LINE_END) {
    for (uint32_t cj = 0; cj < cj_fail; cj++) {
      const kk_transition_subelem_t pi_cj_weight = 
        tex1Dfetch<kk_transition_subelem_t>(tex_transitions, 
                                            fail_base_index + cj + 1);
      if (pi_cj_weight == 0) continue;
      kk_place_subelem_t* const marking_ptr = 
        KK_PLACE_GET_SUBELEM_PTR(marking_array, fail_pi_ref, cj, 
                                glob_color_degree);
      atomicAdd(marking_ptr, pi_cj_weight);
    }
  }

  return false;
}

/**
 * pt_check_places - checks if a transition in a PT-net is activated
 * @tex_transitions: texture object for (maybe fragmented) transition data
 * @in_element0: first element of the preset line of the transition
 * @marking_array: pointer to the marking array
 * @transition_line_elems: number of elements one transition line (pre/postset)
 * @transition_name: name of the transition (for debugging)
 *
 * This function behaves exactly like the kk_check_places function,
 * without multiple token types built into the marking array.
 * 
 * Context: This function is lockless and is to be recalled by the simulator
 * 
 * Return: true if the transition is activated, false otherwise.
 */
static __device__ __forceinline__ bool pt_check_places(
  cudaTextureObject_t tex_transitions,
  const uint32_t      in_element0,
  pt_place_t*         marking_array,
  uint32_t            transition_line_elems
#if defined(DEBUG)
  , char*             transition_name
#endif
) {
  uint32_t input_places_reserved = 0;
  for (uint32_t i = 0; i < transition_line_elems; i++) {
    pt_transition_elem_t transition_elem =
      tex1Dfetch<pt_transition_elem_t>(tex_transitions, in_element0 + i);
    pt_transition_elem_placepart_t placepart =
      PT_TRANSITION_ELEM_GET_PLACE(transition_elem);
    pt_transition_elem_weightpart_t weightpart =
      PT_TRANSITION_ELEM_GET_WEIGHT(transition_elem);
    if (placepart == PT_TRANSITION_LINE_END) return true;
    pt_place_t current_marking = atomicAdd(&marking_array[placepart], 0);
    while (current_marking >= weightpart) {
      pt_place_t previous_marking = atomicCAS(&marking_array[placepart],
                                              current_marking, 
                                              current_marking - weightpart);
      if (previous_marking == current_marking) {
        input_places_reserved++;
        goto next_input_place;
      }
      current_marking = previous_marking;
    }
    goto rollback;
  next_input_place:;
  }
  return true;

rollback:
  for (uint32_t j = 0; j < input_places_reserved; j++) {
    pt_transition_elem_t transition_elem_to_revert =
      tex1Dfetch<pt_transition_elem_t>(tex_transitions, in_element0 + j);

    atomicAdd(
      &marking_array[PT_TRANSITION_ELEM_GET_PLACE(transition_elem_to_revert)],
      PT_TRANSITION_ELEM_GET_WEIGHT(transition_elem_to_revert)
    );
  }
  return false;
}

/**
 * be_check_places - checks if a transition in a BE-net is activated
 * @tex_transitions: texture object for (maybe fragmented) transition data
 * @in_element0: first element of the preset line of the transition
 * @out_element0: first element of the postset line of the transition
 * @marking_array: pointer to the marking array (const, only read)
 * @locks: pointer to the locks array, used to lock places
 * @transition_line_elems: number of elements one transition line (pre/postset)
 * @transition_name: name of the transition (for debugging)
 *
 * This function behaves exactly like the pt_check_places function, but uses 
 * locks to ensure that the marking array is not modified, so no tokens are
 * reserved. BE nets can not pre-produce tokens and then reroll if some postset
 * places are not marked, since this would lead to inconsistent localities
 * for transitions neighbouring this transition. 
 * 
 * Context: This function locks the place locks, and if not all locks get
 *          acquired, it rolls back the acquired locks and returns false.
 *          Return state identifies if all locks were acquired or not.
 * 
 * Return: true if the transition is activated, false otherwise.
 */
static __device__ __forceinline__ bool be_check_places(
  cudaTextureObject_t              tex_transitions,
  const uint32_t                   in_element0,
  const uint32_t                   out_element0,
  const be_place_t* __restrict__   marking_array,
  uint32_t*                        locks,
  uint32_t                         transition_line_elems
#if defined(DEBUG)
  , char*                          transition_name
#endif
) {
  uint32_t input_locks_held = 0;
  uint32_t output_locks_held = 0;

  for (uint32_t i = 0; i < transition_line_elems; i++) {
    be_transition_elem_t transition_elem = 
      tex1Dfetch<be_transition_elem_t>(tex_transitions, in_element0 + i);
    if (transition_elem == BE_TRANSITION_LINE_END) break;
    if (!try_lock(&locks[transition_elem])) goto free_input_locks;
    input_locks_held++;
    if (!marking_array[transition_elem]) goto free_input_locks;
  }
  /****************************************************** now for place_out ***/
  for (uint32_t i2 = 0; i2 < transition_line_elems; i2++) {    
    be_transition_elem_t transition_elem = 
      tex1Dfetch<be_transition_elem_t>(tex_transitions, out_element0 + i2);
    if (transition_elem == BE_TRANSITION_LINE_END) break;
    if (!try_lock(&locks[transition_elem])) goto free_output_locks;
    output_locks_held++;
    if (marking_array[transition_elem]) goto free_output_locks;
  }
  return true;

free_output_locks:
  for (uint32_t j = 0; j < output_locks_held; j++) {
    be_transition_elem_t transition_elem = 
      tex1Dfetch<be_transition_elem_t>(tex_transitions, out_element0 + j);
    release_lock(&locks[transition_elem]);
  }

free_input_locks:
  for (uint32_t j = 0; j < input_locks_held; j++) {
    be_transition_elem_t transition_elem = 
      tex1Dfetch<be_transition_elem_t>(tex_transitions, in_element0 + j);
    release_lock(&locks[transition_elem]);
  }
  return false;
}

/**
 * kk_update_markings - updates markings of a transition's locality in a KK-net
 * @tex_transitions: texture object for (maybe fragmented) transition data
 * @out_element0: first element of the postset line of the transition
 * @marking_array: pointer to the marking array to be modified
 * @glob_color_degree: the global color degree of the net (max cd size)
 * @transition_line_elems: number of elements in transition line (pre/postset)
 * @transition_name: name of the transition (for debugging)
 *
 * This function updates the markings in the marking array under the assumption
 * that the transition is activated. Since the preset places were already
 * reserved, this function only updates the postset markings.
 *
 * Context: This function is lockless and is called after kk_check_places
 *
 * Return: void
 */
static __device__ __forceinline__ void kk_update_markings(
  cudaTextureObject_t tex_transitions,
  const uint32_t      out_element0,
  kk_place_subelem_t* marking_array,
  uint32_t            glob_color_degree,
  uint32_t            transition_line_elems
#if defined(DEBUG)
  , char*             transition_name
#endif
) {
  const uint32_t stride = glob_color_degree + 1;
  for (uint32_t i = 0; i < transition_line_elems;) {
    const uint32_t base_index = out_element0 + i;
    kk_transition_subelem_t pi_ref = 
      tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index);
    if (pi_ref == KK_TRANSITION_LINE_END) break;
    for (uint32_t cj = 0; cj < glob_color_degree; cj++) {
      kk_transition_subelem_t pi_cj_weight = 
        tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index 
                                                             + cj + 1);
      if (pi_cj_weight == 0) continue;
      kk_place_subelem_t* marking_ptr = 
        KK_PLACE_GET_SUBELEM_PTR(marking_array, pi_ref, cj, glob_color_degree);
      atomicAdd(marking_ptr, pi_cj_weight);
    }
    i += stride;
  }
}

/**
 * kk_update_markings - updates markings of a transition's locality in a PT-net
 * @tex_transitions: texture object for (maybe fragmented) transition data
 * @out_element0: first element of the postset line of the transition
 * @marking_array: pointer to the marking array to be modified
 * @transition_line_elems: number of elements in transition line (pre/postset)
 * @transition_name: name of the transition (for debugging)
 *
 * This function behaves exactly like the kk_update_markings function, but
 * without multiple token types built into the marking array.
 *
 * Context: This function is lockless and is called after pt_check_places
 *
 * Return: void
 */
static __device__ __forceinline__ void pt_update_markings(
  cudaTextureObject_t tex_transitions,
  const uint32_t      out_element0,
  pt_place_t*         marking_array,
  uint32_t            transition_line_elems
#if defined(DEBUG)
  , char*             transition_name
#endif
) {
  for (uint32_t i = 0; i < transition_line_elems; i++) {
    pt_transition_elem_t transition_elem = 
      tex1Dfetch<pt_transition_elem_t>(tex_transitions, out_element0 + i);
    pt_transition_elem_weightpart_t weightpart = 
      PT_TRANSITION_ELEM_GET_WEIGHT(transition_elem);
    pt_transition_elem_placepart_t placepart = 
      PT_TRANSITION_ELEM_GET_PLACE(transition_elem);
    if (placepart == PT_TRANSITION_LINE_END) break;
    atomicAdd(&marking_array[placepart], weightpart);
    /* NO UNLOCK! */
  }
}

/**
 * be_update_markings_rel_locks - updates markings of a transition's locality in
 *                                a BE-net
 * @tex_transitions: texture object for (maybe fragmented) transition data
 * @in_element0: first element of the preset line of the transition
 * @out_element0: first element of the postset line of the transition
 * @marking_array: pointer to the marking array to be modified
 * @locks: pointer to the locks array, used to unlock places
 * @transition_line_elems: number of elements in transition line (pre/postset)
 * @transition_name: name of the transition (for debugging)
 *
 * This function behaves exactly like the pt_update_markings function, but for
 * boolean and not counter-type markings. It updates the markings under the 
 * assumption that the transition is activated, which means that the locality
 * is locked. Therefore, after the update, the locks are released.
 *
 * Context: This function needs a locked locality and is called after 
 *          be_check_places. It releases the locks after updating the markings.
 *
 * Return: void
 */
template<bool update_markings>
static __device__ __forceinline__ void be_update_markings_rel_locks(
  cudaTextureObject_t tex_transitions,
  const uint32_t      in_element0,
  const uint32_t      out_element0,
  be_place_t*         marking_array,
  uint32_t*           locks,
  uint32_t            transition_line_elems
#if defined(DEBUG)
  , char*             transition_name
#endif
) {
  for (uint32_t i = 0; i < transition_line_elems; i++) {
    be_transition_elem_t transition_elem = 
      tex1Dfetch<be_transition_elem_t>(tex_transitions, in_element0 + i);
    if (transition_elem == BE_TRANSITION_LINE_END) break;
    if constexpr (update_markings) marking_array[transition_elem] = 0;
    release_lock(&locks[transition_elem]);
  }

  for (uint32_t i = 0; i < transition_line_elems; i++) {
    be_transition_elem_t transition_elem = 
      tex1Dfetch<be_transition_elem_t>(tex_transitions, out_element0 + i);
    if (transition_elem == BE_TRANSITION_LINE_END) break;
    if constexpr (update_markings) marking_array[transition_elem] = BE_MARKED;
    release_lock(&locks[transition_elem]);
  }

  /* release edits which were non-atomic */
  __threadfence();
}

/**
 * transition_kernel - Simulator kernel for petri nets (BE, PT, KK)
 * @net_type_t: Template - Type of the net (e.g., be_net_t, pt_net_t, kk_net_t)
 * @dev_dev_props: Device properties for the current CUDA device of user
 *                 living on device memory
 * @net_data: Net data structure containing important pointers to the net data
 *            and metadata, copied from host
 * @run_cond: Either a pointer to the number of steps left in the simulation,
 *             or a pointer to a flag, that indicates wait, terminate and run,
 *              living on device memory
 * @dev_placelocks: Pointer to the place locks array, used to lock BE places,
 *                  0 otherwise, living on device
 * @t_tex: size annotated array of different texture objects, segmented from 
 *         major transition data if too big
 * @min_transitions_in_tex: Minimum number of transitions in each texture object
 *                          to track which transition is in which tex fragment
 * 
 * This kernel simulates the firing of transitions in a petri net. One thread
 * handles one transition and works on its locality. It shares the 
 * implementation for the different net types (BE, PT, KK) by using 
 * compile-time polymorphism.
 * The kernel checks if a transition is fireable, either by
 * locking in BE-nets or reserving the preset-tokens in PT and KK-nets. Then,
 * the markings are updated accordingly and locks are released if necessary.
 * For locking or reservations collisions, exponential backoff is used, which
 * mitigates global orchestration by the (slower) host.
 *
 * Context: This kernel is called from the host with the number of steps to
 *          simulate, and it runs until all steps are consumed.
 *          It is to be called with more threads than transitions for occupancy
 *          calculations.
 *          The kernel modifies the marking array where 
 *          net_data.net.dev_marking_arr points to.
 *
 * Return: void
 */
template<bool run_cond_is_steps, typename net_type_t>
static __global__ void transition_kernel(
  device_prop_t*           dev_dev_props,
  net_data_t<net_type_t>   net_data,
  int32_t*                 run_cond,
  uint32_t*                dev_placelocks,
  fragm_transitions_tex_t  t_tex
) {
  uint32_t transition_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (transition_idx >= net_data.transitions) return; 

#ifdef DEBUG
  char transition_name[64];
  char *p = transition_name;
  *p++ = 'T';
  p += int_to_str(p, transition_idx);
  *p++ = '-';
  *p++ = '(';
  *p++ = 'B';
  p += int_to_str(p, blockIdx.x);
  *p++ = ')';
  *p = '\0';
#endif

  cudaTextureObject_t tex_transitions = 
    t_tex.arr[transition_idx / t_tex.min_transitions_per_tex];
  uint32_t in_tex_t_id = transition_idx % t_tex.min_transitions_per_tex;

  const uint32_t transition_offset = in_tex_t_id 
                                     * (2 * net_data.transition_line_elems);
  const uint32_t in_element0 = transition_offset;
  const uint32_t out_element0 = in_element0 + net_data.transition_line_elems;

  uint16_t backoff = MIN_BACKOFF;

  while (true) {
    if constexpr (run_cond_is_steps) {
      if (atomicAdd(run_cond, 0) <= 0) break;
    } else {
      if (atomicAdd(run_cond, 0) == SIMULATION_TERMINATE) { break; }
      else if (atomicAdd(run_cond, 0) == SIMULATION_WAIT) {
        __nanosleep(SIMULATION_WAIT_NANOS);
        continue;
      }
    }

    /********************************************************** try to lock ***/
    bool got_locks;
    if constexpr (SAME_TYPE(net_type_t, be_net_t)) {
      got_locks = be_check_places(
        tex_transitions, in_element0, out_element0,
        net_data.net.dev_marking_arr, dev_placelocks, 
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    } else if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
      /* this has no locks */
      got_locks = pt_check_places(
        tex_transitions, in_element0,
        net_data.net.dev_marking_arr,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    } else if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
      /* this has no locks */
      got_locks = kk_check_places(
        tex_transitions, in_element0,
        net_data.net.dev_marking_arr, net_data.glob_color_degree,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    }

    if (!got_locks) {
      uint32_t jitter = (threadIdx.x * 797) % backoff;
      __nanosleep(backoff + jitter);
      backoff = min(backoff << 1, MAX_BACKOFF);
      continue;
    }
    backoff = MIN_BACKOFF;

    /********************************************************* now fireable ***/

    bool sim_ended;
    if constexpr (run_cond_is_steps) {
      sim_ended = atomicSub(run_cond, 1) <= 0;
    } else {
      sim_ended = atomicAdd(run_cond, 0) == SIMULATION_TERMINATE; 
    }

    if (sim_ended) {

      /* we reserved tokens but the simulation ended. We MUST roll them back */
      if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
        const uint32_t stride = net_data.glob_color_degree + 1;
        for (uint32_t i = 0; i < net_data.transition_line_elems;) {
          const uint32_t base_index = in_element0 + i;
          kk_transition_subelem_t pi_ref = 
            tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index);
          if (pi_ref == KK_TRANSITION_LINE_END) break;
          for (uint32_t cj = 0; cj < net_data.glob_color_degree; cj++) {
            kk_transition_subelem_t pi_cj_weight = 
              tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index 
                                                                   + cj + 1);
            if (pi_cj_weight == 0) continue;
            kk_place_subelem_t* marking_ptr = 
              KK_PLACE_GET_SUBELEM_PTR(net_data.net.dev_marking_arr, pi_ref, cj, 
                                       net_data.glob_color_degree);
            atomicAdd(marking_ptr, pi_cj_weight);
          }
          i += stride;
        }
      } else if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
        for (uint32_t j = 0; j < net_data.transition_line_elems; j++) {
          pt_transition_elem_t elem = 
            tex1Dfetch<pt_transition_elem_t>(tex_transitions, in_element0 + j);
          pt_transition_elem_placepart_t place = 
            PT_TRANSITION_ELEM_GET_PLACE(elem);
          if (place == PT_TRANSITION_LINE_END) break;
          atomicAdd(&net_data.net.dev_marking_arr[place],
                    PT_TRANSITION_ELEM_GET_WEIGHT(elem));
        }
      }
      return;
    }

    if constexpr (SAME_TYPE(net_type_t, be_net_t)) {
      be_update_markings_rel_locks<true>(
        tex_transitions, in_element0, out_element0,
        net_data.net.dev_marking_arr,
        dev_placelocks,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    } else if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
      /* this has no locks */
      pt_update_markings(
        tex_transitions, out_element0,
        net_data.net.dev_marking_arr,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    } else if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
      /* this has no locks */
      kk_update_markings(
        tex_transitions, out_element0,
        net_data.net.dev_marking_arr, net_data.glob_color_degree,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    }

    __nanosleep(MIN_BACKOFF);
    backoff = MIN_BACKOFF;
  }
}

template <typename net_type_t>
static __global__ void sync_transition_kernel(
  device_prop_t*           dev_dev_props,
  net_data_t<net_type_t>   net_data,
  int32_t*                 run_cond,
  uint32_t*                dev_placelocks,
  fragm_transitions_tex_t  t_tex,
  uint32_t*                managed_uplink_msg_ptr,
  uint32_t*                managed_downlink_msg_ptr
) {
  uint32_t transition_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (transition_idx >= net_data.transitions) return; 

  uint8_t transition_type;
  uint32_t special_local_id = 0xFFFFFFFF;
  if (transition_idx < net_data.transitions  
                       - net_data.zmq_metadata.zmq_uplinks 
                       - net_data.zmq_metadata.zmq_downlinks) {
    transition_type = SYNC_TRANSITION_LOCAL; /* local transition */
  } else if (transition_idx < net_data.transitions 
                              - net_data.zmq_metadata.zmq_downlinks) {
    transition_type = SYNC_TRANSITION_UPLINK;
    special_local_id = transition_idx - (net_data.transitions
                                      - net_data.zmq_metadata.zmq_uplinks
                                      - net_data.zmq_metadata.zmq_downlinks);
  } else {
    transition_type = SYNC_TRANSITION_DOWNLINK;
    special_local_id = transition_idx - (net_data.transitions
                                   - net_data.zmq_metadata.zmq_downlinks);
  }

#ifdef DEBUG
  char transition_name[64];
  char *p = transition_name;
  *p++ = 'T';
  p += int_to_str(p, transition_idx);
  *p++ = '-';
  *p++ = '(';
  *p++ = 'B';
  p += int_to_str(p, blockIdx.x);
  *p++ = ')';
  *p = '\0';
#endif

  cudaTextureObject_t tex_transitions = 
    t_tex.arr[transition_idx / t_tex.min_transitions_per_tex];
  uint32_t in_tex_t_id = transition_idx % t_tex.min_transitions_per_tex;

  const uint32_t transition_offset = in_tex_t_id 
                                     * (2 * net_data.transition_line_elems);
  const uint32_t in_element0 = transition_offset;
  const uint32_t out_element0 = in_element0 + net_data.transition_line_elems;

  uint16_t backoff = MIN_BACKOFF;

  uint32_t fetched_in0 = 
    tex1Dfetch<uint32_t>(tex_transitions, in_element0);
  uint32_t fetched_out0 =
    tex1Dfetch<uint32_t>(tex_transitions, out_element0);

  DEBUG_PRINTF("[sync_kernel] transition_idx=%u, type=%u, "
               "special_local_id=%u, in0=%u, out0=%u\n",
                transition_idx, transition_type, special_local_id,
                fetched_in0, fetched_out0);

  while (true) {
    if (atomicAdd(run_cond, 0) == SIMULATION_TERMINATE) { break; }
    else if (atomicAdd(run_cond, 0) == SIMULATION_WAIT) {
      __nanosleep(SIMULATION_WAIT_NANOS);
      continue;
    }
    /********************************************************** try to lock ***/
    bool got_locks;

    if (transition_type == SYNC_TRANSITION_UPLINK 
        && atomicAdd(&managed_uplink_msg_ptr[special_local_id], 0) 
            >= ZMQ_TRANSITION_MSG_ROLLBACK) continue;

    if constexpr (SAME_TYPE(net_type_t, be_net_t)) {
      got_locks = be_check_places(
        tex_transitions, in_element0, out_element0,
        net_data.net.dev_marking_arr, dev_placelocks, 
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    } else if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
      /* this has no locks */
      got_locks = pt_check_places(
        tex_transitions, in_element0,
        net_data.net.dev_marking_arr,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    } else if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
      /* this has no locks */
      got_locks = kk_check_places(
        tex_transitions, in_element0,
        net_data.net.dev_marking_arr, net_data.glob_color_degree,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    }

    if (transition_type == SYNC_TRANSITION_UPLINK) {
      if (got_locks) {
        atomicOr(&managed_uplink_msg_ptr[special_local_id],
                 ZMQ_TRANSITION_DOWNLINK_PATCH_FIRE);
        DEBUG_PRINTF("[sync_kernel][uplink] special_local_id=%u: PATCH_FIRE\n",
                    special_local_id);
      } else {
        atomicOr(&managed_uplink_msg_ptr[special_local_id],
                 ZMQ_TRANSITION_DOWNLINK_PATCH_ROLLBACK);
        DEBUG_PRINTF("[sync_kernel][uplink] special_local_id=%u: "
                     "PATCH_ROLLBACK\n", special_local_id);
      }
    } else if (transition_type == SYNC_TRANSITION_DOWNLINK && got_locks) {
      /* step 1: set managed_downlink_msg_ptr and wait for answer */
      atomicExch(&managed_downlink_msg_ptr[special_local_id],
          special_local_id + net_data.zmq_metadata.zmq_target_uplink_startid);
      DEBUG_PRINTF("[sync_kernel][downlink] special_local_id=%u: waiting for "
                   "answer, set to %u\n", special_local_id,
                   managed_downlink_msg_ptr[special_local_id]);

      while (true) {
        uint32_t x = atomicAdd(&managed_downlink_msg_ptr[special_local_id], 0);
        if (x == ZMQ_TRANSITION_MSG_ROLLBACK
            || x == ZMQ_TRANSITION_MSG_FIRE) break; 
      }

      DEBUG_PRINTF("[sync_kernel][downlink] special_local_id=%u: answer %x\n",
                   special_local_id,managed_downlink_msg_ptr[special_local_id]);
      if (atomicAdd(&managed_downlink_msg_ptr[special_local_id], 0)
          == ZMQ_TRANSITION_MSG_ROLLBACK) {
        DEBUG_PRINTF("[sync_kernel][downlink] special_local_id=%u: ROLLBACK, "
                     "rolling back tokens\n", special_local_id);

        /* we reserved tokens but the simulation ended. We MUST roll them back*/
        if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
          const uint32_t stride = net_data.glob_color_degree + 1;
          for (uint32_t i = 0; i < net_data.transition_line_elems;) {
            const uint32_t base_index = in_element0 + i;
            kk_transition_subelem_t pi_ref = 
              tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index);
            if (pi_ref == KK_TRANSITION_LINE_END) break;
            for (uint32_t cj = 0; cj < net_data.glob_color_degree; cj++) {
              kk_transition_subelem_t pi_cj_weight = 
                tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index 
                                                                    + cj + 1);
              if (pi_cj_weight == 0) continue;
              kk_place_subelem_t* marking_ptr = 
                KK_PLACE_GET_SUBELEM_PTR(net_data.net.dev_marking_arr, pi_ref,
                                         cj, net_data.glob_color_degree);
              atomicAdd(marking_ptr, pi_cj_weight);
            }
            i += stride;
          }
        } else if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
          for (uint32_t j = 0; j < net_data.transition_line_elems; j++) {
            pt_transition_elem_t elem = 
              tex1Dfetch<pt_transition_elem_t>(tex_transitions, in_element0 +j);
            pt_transition_elem_placepart_t place = 
              PT_TRANSITION_ELEM_GET_PLACE(elem);
            if (place == PT_TRANSITION_LINE_END) break;
            atomicAdd(&net_data.net.dev_marking_arr[place],
                      PT_TRANSITION_ELEM_GET_WEIGHT(elem));
          }
        } else if constexpr (SAME_TYPE(net_type_t, be_net_t)) {
          be_update_markings_rel_locks<false>(
            tex_transitions, in_element0, out_element0,
            net_data.net.dev_marking_arr,
            dev_placelocks,
            net_data.transition_line_elems
#if defined(DEBUG)
            , transition_name
#endif
          );
        }
      }

      /* else resume, we have the locks ! */
      atomicExch(&managed_downlink_msg_ptr[special_local_id], 
                 ZMQ_TRANSITION_MSG_NONE);
      DEBUG_PRINTF("[sync_kernel][downlink] special_local_id=%u: "
                   "reset to NONE\n", special_local_id);
    }

    if (!got_locks) {
      uint32_t jitter = (threadIdx.x * 797) % backoff;
      __nanosleep(backoff + jitter);
      backoff = min(backoff << 1, MAX_BACKOFF);
      continue;
    }
    backoff = MIN_BACKOFF;

    /********************************************************* now fireable ***/

    bool sim_ended = atomicAdd(run_cond, 0) == SIMULATION_TERMINATE; 

    if (sim_ended) {

      /* we reserved tokens but the simulation ended. We MUST roll them back */
      if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
        const uint32_t stride = net_data.glob_color_degree + 1;
        for (uint32_t i = 0; i < net_data.transition_line_elems;) {
          const uint32_t base_index = in_element0 + i;
          kk_transition_subelem_t pi_ref = 
            tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index);
          if (pi_ref == KK_TRANSITION_LINE_END) break;
          for (uint32_t cj = 0; cj < net_data.glob_color_degree; cj++) {
            kk_transition_subelem_t pi_cj_weight = 
              tex1Dfetch<kk_transition_subelem_t>(tex_transitions, base_index 
                                                                   + cj + 1);
            if (pi_cj_weight == 0) continue;
            kk_place_subelem_t* marking_ptr = 
              KK_PLACE_GET_SUBELEM_PTR(net_data.net.dev_marking_arr, pi_ref, cj, 
                                       net_data.glob_color_degree);
            atomicAdd(marking_ptr, pi_cj_weight);
          }
          i += stride;
        }
      } else if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
        for (uint32_t j = 0; j < net_data.transition_line_elems; j++) {
          pt_transition_elem_t elem = 
            tex1Dfetch<pt_transition_elem_t>(tex_transitions, in_element0 + j);
          pt_transition_elem_placepart_t place = 
            PT_TRANSITION_ELEM_GET_PLACE(elem);
          if (place == PT_TRANSITION_LINE_END) break;
          atomicAdd(&net_data.net.dev_marking_arr[place],
                    PT_TRANSITION_ELEM_GET_WEIGHT(elem));
        }
      }
      return;
    }

    if (transition_type == 0) {
      DEBUG_PRINTF("[sync_kernel][local] transition_idx=%u fired\n", 
                   transition_idx);
    } else if (transition_type == 1) {
      DEBUG_PRINTF("[sync_kernel][uplink] special_local_id=%u fired\n", 
                   special_local_id);
    } else {
      DEBUG_PRINTF("[sync_kernel][downlink] special_local_id=%u fired\n",
                   special_local_id);
    }

    if constexpr (SAME_TYPE(net_type_t, be_net_t)) {
      be_update_markings_rel_locks<true>(
        tex_transitions, in_element0, out_element0,
        net_data.net.dev_marking_arr,
        dev_placelocks,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    } else if constexpr (SAME_TYPE(net_type_t, pt_net_t)) {
      /* this has no locks */
      pt_update_markings(
        tex_transitions, out_element0,
        net_data.net.dev_marking_arr,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    } else if constexpr (SAME_TYPE(net_type_t, kk_net_t)) {
      /* this has no locks */
      kk_update_markings(
        tex_transitions, out_element0,
        net_data.net.dev_marking_arr, net_data.glob_color_degree,
        net_data.transition_line_elems
#if defined(DEBUG)
        , transition_name
#endif
      );
    }

#if defined(DEBUG)
    // print marking in hex
    printf(" Marking: ");
    for (uint32_t i = 0; i < net_data.places; i++) {
      printf("%u ", net_data.net.dev_marking_arr[i]);
    }
    printf("\n");
#endif

    __nanosleep(MIN_BACKOFF);
    backoff = MIN_BACKOFF;
  }
}


/* -------------------------------------------------------------------------- */
/* ------------------------------- main part -------------------------------- */
/* --------------------------------- host ----------------------------------- */
/* -------------------------------------------------------------------------- */

/* This variable is only set if a simulation is about to be started without */
/* a step counter                                                           */
/* attention: this gets optimized away very quickly for some functions, esp.*/
/* the zmq thread functions, therefore volatile                             */
volatile static int32_t* managed_ctrlflag_ptr = 0;

/* used for time calculation before kernel execution and during */
struct timespec start, prekernel, postkernel;

/* variables shared for zmq network setup */
/* they are set in main                   */
static bool remote_mode = false;
static char* recv_port = 0;
static char* send_addr = 0;
static zmq::context_t* global_zmq_ctx = 0;

__forceinline__ static void setup_pull_socket(zmq::socket_t& sock) {
  sock.set(zmq::sockopt::linger, 0);
  sock.set(zmq::sockopt::rcvhwm, 100000);
  sock.set(zmq::sockopt::rcvbuf, 16777216);
  sock.set(zmq::sockopt::rcvtimeo, 1000);
}

__forceinline__ static void setup_push_socket(zmq::socket_t& sock) {
  sock.set(zmq::sockopt::linger, 0);
  sock.set(zmq::sockopt::sndhwm, 100000);
  sock.set(zmq::sockopt::sndbuf, 16777216);
  sock.set(zmq::sockopt::sndtimeo, 1000);
}


static void zmq_uplink_pull_worker(
  zmq::context_t*      ctx,
  uint32_t*            uplink_msg_ptr,
  uint32_t             uplinks,
  uint32_t             uplink_globalstartid
) {
  uint64_t recv_messages = 0;
  zmq::socket_t pull(*ctx, ZMQ_PULL);
  setup_pull_socket(pull);
  DEBUG_PRINTF("[uplink_pull_worker] Binding to first port of %s...\n",
               recv_port);
  zmq_bind_from_pair1(pull, recv_port);
  uplink_downlink_msg_t msg;
  struct timespec sendtime;

  while (*managed_ctrlflag_ptr != SIMULATION_TERMINATE) {
    if (*managed_ctrlflag_ptr == SIMULATION_WAIT) {
      usleep(1000);
      continue;
    }
    zmq::message_t zmq_msg(sizeof(uplink_downlink_msg_t));
    auto result = pull.recv(zmq_msg, zmq::recv_flags::none);
    if (result && zmq_msg.size() == sizeof(uplink_downlink_msg_t)) {
      memcpy(&msg, zmq_msg.data(), sizeof(uplink_downlink_msg_t));
      DEBUG_PRINTF("[uplink_pull_worker] Received message: global_uplink_id=%u,"
                   " global_downlink_id=%u\n", msg.global_uplink_id,
                   msg.global_downlink_id);
      recv_messages++;
      if (msg.global_uplink_id >= uplink_globalstartid && 
          msg.global_uplink_id < uplink_globalstartid + uplinks) {
        uint32_t local_uplink_idx = msg.global_uplink_id - uplink_globalstartid;
        uplink_msg_ptr[local_uplink_idx] = msg.global_downlink_id;
        clock_gettime(CLOCK_TYPE, &sendtime);
        double t = timespec_diff_ms(&start, &sendtime);
        DEBUG_PRINTF("[uplink_pull_worker] Updated uplink_msg_ptr[%u]=%u at "
                     "%.9f ms\n", local_uplink_idx, msg.global_downlink_id, t);
      } else {
        DEBUG_PRINTF("[uplink_pull_worker] ERROR: Received invalid uplink_id=%u"
                     ", expected range [%u, %u)\n", msg.global_uplink_id,
                      uplink_globalstartid, uplink_globalstartid + uplinks);
      }
    }
  }

  printf("[uplink_pull_worker] Stopping ZMQ uplink worker, received %lu "
         "messages...\n", recv_messages);
  try {
    pull.close();
  } catch (const zmq::error_t& e) {
    printf("[uplink_pull_worker] Error closing socket: %s\n", e.what());
  } catch (...) {
    printf("[uplink_pull_worker] Unknown exception, closing socket\n");
  }
}


static void zmq_uplink_push_worker(
  zmq::context_t*      ctx,
  uint32_t*            uplink_msg_ptr,
  uint32_t             uplinks
) {
  uint64_t sent_messages = 0;
  zmq::socket_t push(*ctx, ZMQ_PUSH);
  setup_push_socket(push);
  DEBUG_PRINTF("[uplink_push_worker] Binding to second port of %s...\n",
               recv_port);
  zmq_bind_from_pair2(push, recv_port);

  uplink_downlink_msg_t msg;
  struct timespec sendtime;

  while (*managed_ctrlflag_ptr != SIMULATION_TERMINATE) {
    if (*managed_ctrlflag_ptr == SIMULATION_WAIT) {
      usleep(1000);
      continue;
    }
    for (uint32_t i = 0; i < uplinks; i++) {
      uint32_t current_msg = uplink_msg_ptr[i];
      if (current_msg == ZMQ_TRANSITION_MSG_NONE) {
        continue;
      }
      zmq::message_t zmq_msg(sizeof(uplink_downlink_msg_t));
      if (current_msg & ZMQ_TRANSITION_DOWNLINK_PATCH_MASK) {
        uint32_t target_downlink_id = current_msg 
                                      & ~ZMQ_TRANSITION_DOWNLINK_PATCH_MASK;
        bool is_fire = (current_msg & ZMQ_TRANSITION_DOWNLINK_PATCH_FIRE) != 0;
        DEBUG_PRINTF("[uplink_push_worker] Preparing to send: local_uplink=%u, "
                     "target_downlink_id=%u, is_fire=%d\n", i, 
                     target_downlink_id, is_fire);
        msg.global_downlink_id = target_downlink_id;
        msg.msg = is_fire ? ZMQ_TRANSITION_MSG_FIRE:ZMQ_TRANSITION_MSG_ROLLBACK;
        uplink_msg_ptr[i] = ZMQ_TRANSITION_MSG_NONE;
        memcpy(zmq_msg.data(), &msg, sizeof(uplink_downlink_msg_t));
        try {
          DEBUG_PRINTF("[uplink_push_worker] Sending message to downlink_id=%u,"
                      " msg=%s\n", msg.global_downlink_id, 
                      is_fire ? "FIRE" : "ROLLBACK");
          push.send(zmq_msg, zmq::send_flags::dontwait);
          clock_gettime(CLOCK_TYPE, &sendtime);
          double t = timespec_diff_ms(&start, &sendtime);
          DEBUG_PRINTF("[uplink_push_worker] Sent at %.9f ms\n", t);
          sent_messages++;
        } catch (const zmq::error_t& e) {
          DEBUG_PRINTF("[uplink_push_worker] ZMQ exception: %s\n", e.what());
          if (e.num() == ETERM) {
            DEBUG_PRINTF("[uplink_push_worker] Context terminated, exiting.\n");
            break;
          }
        } catch (...) {
          DEBUG_PRINTF("[uplink_push_worker] Unknown exception caught\n");
        }
      }
    }
  }

  printf("[uplink_push_worker] Stopping ZMQ uplink worker, sent %lu messages"
         "...\n", sent_messages);
  try {
    push.close();
  } catch (const zmq::error_t& e) {
    printf("[uplink_push_worker] Error closing socket: %s\n", e.what());
  } catch (...) {
    printf("[uplink_push_worker] Unknown exception, closing socket\n");
  }
}


static void zmq_downlink_pull_worker(
  zmq::context_t* ctx,
  uint32_t* downlink_msg_ptr,
  uint32_t downlinks,
  uint32_t downlink_globalstartid
) {
  uint64_t recv_messages = 0;
  zmq::socket_t pull(*ctx, ZMQ_PULL);
  setup_pull_socket(pull);
  DEBUG_PRINTF("[downlink_pull_worker] Connecting to first addr of %s...\n",
               send_addr);
  zmq_connect_from_pair1(pull, send_addr);

  uplink_downlink_msg_t msg;

  struct timespec sendtime;

  while (*managed_ctrlflag_ptr != SIMULATION_TERMINATE) {
    if (*managed_ctrlflag_ptr == SIMULATION_WAIT) {
      usleep(1000);
      continue;
    }
    zmq::message_t zmq_msg(sizeof(uplink_downlink_msg_t));
    auto result = pull.recv(zmq_msg, zmq::recv_flags::none);
    if (result && zmq_msg.size() == sizeof(uplink_downlink_msg_t)) {
      memcpy(&msg, zmq_msg.data(), sizeof(uplink_downlink_msg_t));
      DEBUG_PRINTF("[downlink_pull_worker] Received message: "
                   "global_downlink_id=%u, msg=%u\n", 
                   msg.global_downlink_id, msg.msg);
      recv_messages++;
      if (msg.global_downlink_id >= downlink_globalstartid && 
          msg.global_downlink_id < downlink_globalstartid + downlinks) {
        uint32_t local_index = msg.global_downlink_id - downlink_globalstartid;
        downlink_msg_ptr[local_index] = msg.msg;
        clock_gettime(CLOCK_TYPE, &sendtime);
        double t = timespec_diff_ms(&start, &sendtime);
        DEBUG_PRINTF("[downlink_pull_worker] Updated downlink_msg_ptr[%u]=%u"
                     " at %.9f ms\n", local_index, msg.msg, t);
      } else {
        DEBUG_PRINTF("[downlink_pull_worker] ERROR: Received invalid "
                     " downlink_id=%u, expected range [%u, %u)\n",
                     msg.global_downlink_id, downlink_globalstartid,
                     downlink_globalstartid + downlinks);
      }
    }
  }

  printf("[downlink_pull_worker] Stopping ZMQ downlink worker, received %lu "
         "messages...\n", recv_messages);
  try {
    pull.close();
  } catch (const zmq::error_t& e) {
    printf("[downlink_pull_worker] Error closing socket: %s\n", e.what());
  } catch (...) {
    printf("[downlink_pull_worker] Unknown exception, closing socket\n");
  }
}


static void zmq_downlink_push_worker(
  zmq::context_t* ctx,
  uint32_t* downlink_msg_ptr,
  uint32_t downlinks,
  uint32_t downlink_globalstartid
) {
  uint64_t sent_messages = 0;
  zmq::socket_t push(*ctx, ZMQ_PUSH);
  setup_push_socket(push);
  DEBUG_PRINTF("[downlink_push_worker] Connecting to second addr of %s...\n",
              send_addr);
  zmq_connect_from_pair2(push, send_addr);

  uplink_downlink_msg_t msg;

  struct timespec sendtime;

  while (*managed_ctrlflag_ptr != SIMULATION_TERMINATE) {
    if (*managed_ctrlflag_ptr == SIMULATION_WAIT) {
      usleep(1000);
      continue;
    }
    for (uint32_t i = 0; i < downlinks; i++) {
      zmq::message_t zmq_msg(sizeof(uplink_downlink_msg_t));
      uint32_t current_msg = downlink_msg_ptr[i];
      if (current_msg < ZMQ_TRANSITION_MSG_ROLLBACK 
          && current_msg != ZMQ_TRANSITION_MSG_NONE) {
        msg.global_downlink_id = downlink_globalstartid + i;
        msg.global_uplink_id = current_msg;
        DEBUG_PRINTF("[downlink_push_worker] Preparing to send: "
                     "local_downlink=%u, target_uplink_id=%u\n", i,current_msg);
        memcpy(zmq_msg.data(), &msg, sizeof(uplink_downlink_msg_t));
        try {
          DEBUG_PRINTF("[downlink_push_worker] Sending request: downlink_id=%u "
                       "wants uplink_id=%u\n", 
                       msg.global_downlink_id, msg.global_uplink_id);
          auto result = push.send(zmq_msg, zmq::send_flags::dontwait);
          clock_gettime(CLOCK_TYPE, &sendtime);
          double t = timespec_diff_ms(&start, &sendtime);
          DEBUG_PRINTF("[downlink_push_worker] Sent at %.9f ms\n", t);
          sent_messages++;

          if (result.has_value()) {
            downlink_msg_ptr[i] = ZMQ_TRANSITION_MSG_NONE;
            clock_gettime(CLOCK_TYPE, &sendtime);
            double t = timespec_diff_ms(&start, &sendtime);
            DEBUG_PRINTF("[downlink_push_worker] Received at %.9f ms\n", t);
          } else {
            DEBUG_PRINTF("[downlink_push_worker] Send would block, "
                         "retrying next iteration\n");
          }
          
        } catch (const zmq::error_t& e) {
          DEBUG_PRINTF("[downlink_push_worker] ZMQ exception: %s\n", e.what());
          if (e.num() == ETERM) break;
        } catch (...) {
          DEBUG_PRINTF("[downlink_push_worker] Unknown exception caught\n");
        }
      }
    }
  }

  printf("[downlink_push_worker] Stopping ZMQ downlink worker, sent %lu "
         "messages...\n", sent_messages);
  try {
    push.close();
  } catch (const zmq::error_t& e) {
    printf("[downlink_push_worker] Error closing socket: %s\n", e.what());
  } catch (...) {
    printf("[downlink_push_worker] Unknown exception, closing socket\n");
  }
}

__forceinline__ static uint32_t zmq_uplink_downlink_init_threads(
  uint32_t*            managed_uplink_ptr,
  uint32_t*            managed_downlink_ptr,
  uint32_t             uplinks,
  uint32_t             downlinks,
  uint32_t             uplink_globalstartid,
  uint32_t             downlink_globalstartid,
  thread_t***          workers_out
) {
  if (!global_zmq_ctx) {
    global_zmq_ctx = new zmq::context_t(1);
    global_zmq_ctx->set(zmq::ctxopt::io_threads, 4);
    global_zmq_ctx->set(zmq::ctxopt::max_sockets, 65536);
  }

  uint32_t num_threads = 0;
  thread_t** workers = (thread_t**) malloc(sizeof(thread_t*) * 4);
  if (!workers) {
    printf("failed to allocate memory for uplink/downlink worker threads\n");
    return 0;
  }

  try {
    num_threads = 0;
    if (uplinks > 0) {
      
      workers[num_threads++] = new thread_t(
        zmq_uplink_pull_worker,
        global_zmq_ctx,
        managed_uplink_ptr,
        uplinks,
        uplink_globalstartid
      );
      workers[num_threads++] = new thread_t(
        zmq_uplink_push_worker,
        global_zmq_ctx,
        managed_uplink_ptr,
        uplinks
        /* unused: uplink_globalstartid*/
      );
    }
    if (downlinks > 0) {
      workers[num_threads++] = new thread_t(
        zmq_downlink_pull_worker,
        global_zmq_ctx,
        managed_downlink_ptr,
        downlinks,
        downlink_globalstartid
      );
      workers[num_threads++] = new thread_t(
        zmq_downlink_push_worker,
        global_zmq_ctx,
        managed_downlink_ptr,
        downlinks,
        downlink_globalstartid
      );
    }
  } catch (const std::exception& e) {
    printf("failed to create uplink/downlink threads: %s\n", e.what());
    for (uint32_t i = 0; i < num_threads; i++) {
      if (workers[i]) delete workers[i];
    }
    free(workers);
    return 0;
  }

  printf("uplink/downlink threads are ready, starting simulation...\n");
  *workers_out = workers;
  return num_threads;
}


static void zmq_remote_place_input_worker(
  zmq::context_t*      ctx,
  uint32_t*            managed_edge_ptr,
  uint32_t             in_edge_startid
) {
  uint64_t recv_messages = 0;
  zmq::socket_t sock(*ctx, ZMQ_PULL);
  setup_pull_socket(sock);
  DEBUG_PRINTF("[zmq-pull] Binding to port %s...\n", recv_port);
  zmq_bind_listen_port(sock, recv_port);

  forwarded_token_t msg;

  DEBUG_PRINTF("[zmq-pull] Starting to receive messages...\n");

  struct timespec sendtime;

  while (*managed_ctrlflag_ptr != SIMULATION_TERMINATE) {
    if (*managed_ctrlflag_ptr == SIMULATION_WAIT) {
      usleep(1000);
      continue;
    }
    zmq::message_t zmq_msg(sizeof(forwarded_token_t));
    try {
      auto recv = sock.recv(zmq_msg, zmq::recv_flags::none);
      if (!recv) continue;
    } catch (const zmq::error_t& e) {
      if (e.num() == ETERM) {
        printf("[zmq-pull] Context terminated, exiting.\n");
        break;
      }
      printf("[zmq-pull] ZMQ exception caught: %s\n", e.what());
    } catch (...) {
      printf("[zmq-pull] Unknown exception caught\n");
    }
    if (zmq_msg.size() == sizeof(forwarded_token_t)) {
      recv_messages++;
      memcpy(&msg, zmq_msg.data(), sizeof(forwarded_token_t));
      const uint32_t local_id = msg.place_id - in_edge_startid;
      const uint32_t val = msg.val;
      managed_edge_ptr[local_id] += val;
      clock_gettime(CLOCK_TYPE, &sendtime);
      double t = timespec_diff_ms(&start, &sendtime);
      DEBUG_PRINTF("[zmq-pull] Moving %u to global place %u, (local=%u)"
             " (since kernel-start: %.9f ms)\n",
             msg.val, msg.place_id, local_id, t);
    } else {
      DEBUG_PRINTF("[zmq-pull] Received message of unexpected size: %zu\n", 
             zmq_msg.size());
    }
  }

  printf("[zmq-pull] Stopping ZMQ input worker, received %lu messages...\n",
         recv_messages);
  try {
    sock.close();
  } catch (const zmq::error_t& e) {
    printf("[zmq-pull] Error closing socket: %s\n", e.what());
  } catch (...) {
    printf("[zmq-pull] Unknown exception caught while closing socket\n");
  }
}

static void zmq_remote_place_output_worker(
  zmq::context_t*      ctx,
  uint32_t*            managed_edge_ptr,
  uint32_t             elems_in_out_edge,
  uint32_t             out_edge_startid
) {
  uint64_t sent_messages = 0;
  zmq::socket_t sock(*ctx, ZMQ_PUSH);
  setup_push_socket(sock);
  DEBUG_PRINTF("[zmq-push] Connecting to %s...\n", send_addr);
  zmq_connect_addr_port(sock, send_addr);

  forwarded_token_t msg;

  DEBUG_PRINTF("[zmq-push] Starting to send messages...\n");

  struct timespec sendtime;


  while (*managed_ctrlflag_ptr != SIMULATION_TERMINATE) {
    if (*managed_ctrlflag_ptr == SIMULATION_WAIT) {
      usleep(1000);
      continue;
    }
    for (uint32_t i = 0; i < elems_in_out_edge; i++) {
      zmq::message_t zmq_msg(sizeof(forwarded_token_t));
      if (managed_edge_ptr[i] == 0) continue;
      uint32_t val = managed_edge_ptr[i];
      managed_edge_ptr[i] -= val;
      
      msg.val = val;
      msg.place_id = i + out_edge_startid;
      memcpy(zmq_msg.data(), &msg, sizeof(forwarded_token_t));
      
      try {
        sock.send(zmq_msg, zmq::send_flags::dontwait);
        clock_gettime(CLOCK_TYPE, &sendtime);
        double t = timespec_diff_ms(&start, &sendtime);
        DEBUG_PRINTF("[zmq-push] Sent %u to global place %u, (local=%u)"
                     " (since kernel-start: %.9f ms)\n",
                     msg.val, msg.place_id, i, t);
        sent_messages++;
      } catch (const zmq::error_t& e) {
        if (e.num() == ETERM) {
          printf("[zmq-pull] Context terminated, exiting.\n");
          break;
        }
        printf("[zmq-pull] ZMQ exception caught: %s\n", e.what());
      } catch (...) {
        printf("[zmq-pull] Unknown exception caught\n");
      }
    }
  }

  printf("[zmq-push] Stopping ZMQ output worker, sent %lu messages...\n",
         sent_messages);
  try {
    sock.close();
  } catch (const zmq::error_t& e) {
    printf("[zmq-push] Error closing socket: %s\n", e.what());
  } catch (...) {
    printf("[zmq-push] Unknown exception caught while closing socket\n");
  }
}

__forceinline__ static uint32_t zmq_remote_place_init_threads(
  const zmq_metadata_t zmq_metadata,
  uint32_t*            managed_in_edge_ptr,
  uint32_t*            managed_out_edge_ptr,
  const uint32_t       elems_in_out_edge,
  thread_t***          workers_in
) {
  if (!global_zmq_ctx) {
    global_zmq_ctx = new zmq::context_t(1);
    global_zmq_ctx->set(zmq::ctxopt::io_threads, 4);
    global_zmq_ctx->set(zmq::ctxopt::max_sockets, 65536);
  }

  thread_t** workers = (thread_t**) malloc(sizeof(thread_t*) * 2);
  workers[0] = 0;
  workers[1] = 0;
  if (!workers) {
    printf("Failed to allocate memory for worker threads\n");
    return 0;
  }

  try {
    if (recv_port != 0) {
      workers[0] = new thread_t(zmq_remote_place_input_worker, global_zmq_ctx,
                                managed_in_edge_ptr, 
                                zmq_metadata.zmq_in_edge_startid);
    }
    if (send_addr != 0) {
      workers[1] = new thread_t(zmq_remote_place_output_worker, global_zmq_ctx,
                                managed_out_edge_ptr, elems_in_out_edge,
                                zmq_metadata.zmq_out_edge_startid);
    }
  } catch (const std::exception& e) {
    printf("failed to create threads: %s\n", e.what());
    for (uint32_t i = 0; i < 2; i++) {
      if (workers[i]) delete workers[i];
    }
    free(workers);
    return 0;
  }

  printf("all threads are ready, starting simulation...\n");
  *workers_in = workers;
  return 2;
}



__forceinline__ static void zmq_stop_threads(
  thread_t** workers,
  uint32_t   num_threads
) {
  printf("Waiting for ZMQ threads to join...\n");
  usleep(ZMQ_ENDFETCHTIME);
  printf("Stopping ZMQ threads...\n");

  for (uint32_t i = 0; i < num_threads; i++) {
    if (workers[i] == 0) break;
    workers[i]->detach(); /* join does not work here since zmq hangs in poll*/
    delete workers[i];
    workers[i] = 0;
  }

  free(workers);

  if (global_zmq_ctx) {
    global_zmq_ctx->close();
    delete global_zmq_ctx;
    global_zmq_ctx = nullptr;
  }
}

/**
 * alloc_fragmented_transition_tex - allocates a fragmented transition texture
 * @dev_props: pointer to the device properties structure
 * @transition_region_size: size of the transition region in bytes
 * @transition_amount: number of transitions in the region
 * @dev_transitions_arr: pointer to the device memory where transitions are
 *                      stored
 * @net_type: type of the net (BE, PT, KK)
 *
 * This function allocates a fragmented transition texture object that can be
 * used to store transition data for a petri net. It calculates the necessary
 * size and number of textures based on the device properties and the given
 * transition region size.
 *
 * Context: Texture is to be destroyed in the same scope as it was created.
 * 
 * Return: pointer to the allocated fragmented transition texture object or NULL
 *         if allocation failed.
 */
static fragm_transitions_tex_t alloc_fragmented_transition_tex(
  device_prop_t* dev_props,
  uint32_t       transitions_region_size,
  uint32_t       transition_amount,
  uint8_t*       dev_transitions_arr,
  uint32_t       net_type  
) {
  cudaFuncSetCacheConfig(transition_kernel<true, pt_net_t>, 
                         cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(transition_kernel<false, pt_net_t>, 
                         cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(sync_transition_kernel<pt_net_t>, 
                         cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(transition_kernel<true, be_net_t>, 
                         cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(transition_kernel<false, be_net_t>, 
                         cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(sync_transition_kernel<be_net_t>, 
                         cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(transition_kernel<true, kk_net_t>, 
                         cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(transition_kernel<false, kk_net_t>, 
                         cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(sync_transition_kernel<kk_net_t>, 
                         cudaFuncCachePreferL1);

  const uint32_t transition_region_size_padded = 
    ROUND_UP(transitions_region_size, TRANSITION_AREA_PADDING);
  uint32_t transition_size_bytes = transitions_region_size / transition_amount;
  
  uint32_t texamount = (transition_region_size_padded 
                        + dev_props->max_texture_1d_linear - 1) 
                       / dev_props->max_texture_1d_linear;
  if (texamount == 0) texamount = 1;
  uint32_t min_texture_size = ROUND_UP(transition_size_bytes,
                                       TRANSITION_AREA_PADDING);
  if (min_texture_size > dev_props->max_texture_1d_linear) {
    printf("ERROR: Single transition (%u bytes) exceeds max texture size "
           "(%u bytes)\n",
           min_texture_size, dev_props->max_texture_1d_linear);
    return {0,0,0};
  }
  while (texamount > 0) {
    uint32_t normal_part_size = ROUND_UP(transition_region_size_padded
                                         / texamount, TRANSITION_AREA_PADDING);
    uint32_t min_transitions_in_tex = normal_part_size / transition_size_bytes;
    if (min_transitions_in_tex > 0 
        && normal_part_size <= dev_props->max_texture_1d_linear) break;
    texamount++;
  }
  uint32_t normal_part_size = ROUND_UP(transition_region_size_padded /texamount,
                                       TRANSITION_AREA_PADDING);
  uint32_t min_transitions_in_tex = normal_part_size / transition_size_bytes;

  if (min_transitions_in_tex == 0) {
    printf("ERROR: min_transitions_in_tex=0 - texture partitioning failed!\n");
    printf("transition_region_size_padded: %u\n",transition_region_size_padded);
    printf("max_texture: %u\n", dev_props->max_texture_1d_linear);
    printf("texamount: %d\n", texamount);
    printf("normal_part_size: %u\n", normal_part_size);
    printf("transition_size_bytes: %u\n", transition_size_bytes);
    return {0,0,0};
  }

  int32_t theoretical_last_size = transition_region_size_padded 
                                  - (normal_part_size * (texamount-1));
  uint32_t last_part_size;
  if (theoretical_last_size <= 0) {
    --texamount;
    last_part_size = (int32_t) normal_part_size + theoretical_last_size;
  } else {
    last_part_size = (uint32_t) theoretical_last_size;
  }

  fragm_transitions_tex_t t_tex;
  t_tex.amount = texamount;
  t_tex.min_transitions_per_tex = min_transitions_in_tex;
  cudaMallocManaged(&t_tex.arr,
                    texamount * sizeof(cudaTextureObject_t));

  for(uint32_t t = 0; t < texamount; t++) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = 
      (void*) ((uint8_t*) dev_transitions_arr 
               + normal_part_size * t);

    resDesc.res.linear.sizeInBytes = t == t_tex.amount - 1 ? last_part_size 
                                                           : normal_part_size;
    if (net_type == NET_TYPE_BE || net_type == NET_TYPE_SBE) {
      resDesc.res.linear.desc = cudaCreateChannelDesc<be_transition_elem_t>();
    } else if (net_type == NET_TYPE_PT || net_type == NET_TYPE_SPT) {
      resDesc.res.linear.desc = cudaCreateChannelDesc<pt_transition_elem_t>();
    } else if (net_type == NET_TYPE_KK || net_type == NET_TYPE_SKK) {
      resDesc.res.linear.desc =cudaCreateChannelDesc<kk_transition_subelem_t>();
    } else {
      printf("ERROR: unknown net type %u\n", net_type);
      return {0,0,0};
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 0;

    printf("Creating texture %d: bytesize=%zu, transitions>=%d\n",
        t, resDesc.res.linear.sizeInBytes,
        min_transitions_in_tex);

    cudaTextureObject_t tex;
    cudaError_t tex_err = cudaCreateTextureObject(&tex, &resDesc,
                                                  &texDesc, NULL);
    if (tex_err != cudaSuccess) {
      printf("Texture creation FAILED: %s\n", cudaGetErrorString(tex_err));
      printf("Resource Desc: type=%d, ptr=%p, size=%zu, format=%d\n",
              resDesc.resType, resDesc.res.linear.devPtr,
              resDesc.res.linear.sizeInBytes, resDesc.res.linear.desc.f);
      return {0,0,0};
    }
    t_tex.arr[t] = tex;
  } 
  return t_tex;
}

/**
 * destroy_fragmented_transition_tex - destroys a fragmented transition
 * @t_tex: pointer to the fragmented transition texture object
 * 
 * This function destroys the fragmented transition texture object by freeing
 * its array of pointers to the textures and destroying them.
 * 
 * Context: This function is called in the same scope where t_tex was created
 * 
 * Return: true if the texture was successfully destroyed, false if t_tex = 0
 */
static void destroy_fragmented_transition_tex(
  fragm_transitions_tex_t t_tex
) {
  for (uint32_t t = 0; t < t_tex.amount; t++) {
    cudaDestroyTextureObject(t_tex.arr[t]);
  }
  cudaFree(t_tex.arr);
}

/**
 * sim_stop - stops the simulation by setting the control flag
 * 
 * This function sets the control flag to SIMULATION_TERMINATE, which
 * indicates that the simulation should be stopped.
 * 
 * Context: This function is called when the user presses Ctrl+C
 *         It works only while a simulation is running without a step counter,
 *          i.e., when managed_ctrlflag_ptr is set.
 * 
 * Return: true if the simulation was stopped, false if no control flag pointer
 *         was set (e.g., when the simulation is running with a step counter)
 */
bool sim_stop() {
if (managed_ctrlflag_ptr) {
    printf("Terminating simulation...\n");
    *managed_ctrlflag_ptr = SIMULATION_TERMINATE;
    return true;
  } else {
    printf("SIGINT received, but no control flag pointer set...\n");
    return false;
  }
}

/**
 * sim_togglepause - toggles the pause state of the simulation
 * 
 * This function toggles the pause state of the simulation by setting the
 * control flag to SIMULATION_WAIT or SIMULATION_RUN, depending on the current
 * state.
 * 
 * Context: This function is called when the user presses Ctrl+Z
 *          It works only while a simulation is running without a step counter,
 *          i.e., when managed_ctrlflag_ptr is set.
 * 
 * Return: true if the simulation was paused/resumed, false if no control flag
 *         pointer was set (e.g., when the simulation is running with a step
 *         counter)
 */
uint8_t sim_togglepause() {
  if (managed_ctrlflag_ptr) {
    if (*managed_ctrlflag_ptr == SIMULATION_WAIT) {
      printf("Resuming simulation...\n");
      *managed_ctrlflag_ptr = SIMULATION_RUN;
    } else if (*managed_ctrlflag_ptr == SIMULATION_RUN) {
      printf("Pausing simulation...\n");
      *managed_ctrlflag_ptr = SIMULATION_WAIT;
    } else {
      printf("Simulation is not running, cannot toggle pause/resume...\n");
    }
    return true;
  } else {
    printf("SIGTSTP received, but no control flag pointer set...\n");
    return false;
  }
}

/**
 * dissect_netdata_run_kernel - runs the transition kernel on the net data
 * @dev_props: pointer to the device properties structure
 * @managed_filedata_ptr: pointer to the managed file data, which means its
 *                        living in memory both accessible by device and host
 * @netbytes_size: size of the net data in bytes, which is the file-input size
 * @work_on_dev_mem: true if the net data is in dev memory, false otherwise,
 *                  this distinction is for the use in this context, where we
 *                 move the net data into device memory for efficiency, and use
 *                of this function from outside, where the initializer has no
 *               access to CUDA and needs managed memory allocated by some
 *              wrapper.
 * @run_cond: either number of steps left in the simulation, which is 
 *            decremented by the kernel, and retrieved initially through CLI-arg
 *            or a flag that indicates wait, terminate and run, if its 
 *            RUN_COND_CTRLFLAG_IDENTIFIER
 * 
 * This function sets up the device memory for the net data, allocates the
 * fragmented transition texture, and runs the transition kernel.
 * It also handles the case where the net data is in managed memory, in which
 * case it uses the managed memory directly without copying it to device memory.
 * 
 * Context: This function is called under the following assumptions:
 *            - device properties are fetched (dev_props)
 *            - the net data is ready to be processed and lives in memory that
 *              is readable from the host (and device if wanted, mode-specific)
 *            - the overall steps to be executed are parsed from the CLI
 *              or it is interpreted as the control flag (if 
 *                                                 RUN_COND_CTRLFLAG_IDENTIFIER)
 *              (run_cond)
 *
 *            - also, there are some global variables for remote mode, see above
 * 
 * Return: 0 on success, 1 on failure
 */
static uint8_t dissect_netdata_run_kernel(
  device_prop_t* dev_props,
  uint8_t*       managed_filedata_ptr,
  uint32_t       netbytes_size,
  bool           work_on_dev_mem,
  int32_t        run_cond,
  FILE*          marking_out_file
) {
  /* stream setup for async memcpy from managed host mem to GPU */
  cudaStream_t compute_stream, io_stream;
  cudaStreamCreate(&compute_stream);
  cudaStreamCreate(&io_stream);
  
  uint8_t* dev_netarea_ptr;

  if (work_on_dev_mem) {
    cudaMalloc((void**)&dev_netarea_ptr, netbytes_size);
    cudaMemcpyAsync(dev_netarea_ptr, managed_filedata_ptr, netbytes_size,
                    cudaMemcpyHostToDevice, io_stream);
  } else {
    /* use managed host memory directly, no copy needed */
    dev_netarea_ptr = managed_filedata_ptr;
  }

  /* create the data that is worked on by the transition threads */
  net_data_t<void_net_t> net_data = {0,0,0,0,0,0,0,0,0,0,0};
  uint32_t* metadata_ptr = (uint32_t*) managed_filedata_ptr;
  net_data.places = metadata_ptr[0];
  net_data.transitions = metadata_ptr[1];
  net_data.transition_line_elems = metadata_ptr[2];

  /* magic number is last uint32_t in metadata region */
  net_data.magic_number = metadata_ptr[MAGIC_NUMBER_OFFSET];

  uint8_t net_type = net_data.magic_number;
  if (net_data.magic_number != NET_TYPE_PT 
      && net_data.magic_number != NET_TYPE_KK
      && net_data.magic_number != NET_TYPE_SPT
      && net_data.magic_number != NET_TYPE_SKK
      && net_data.magic_number != NET_TYPE_SBE) net_type = NET_TYPE_BE;
  /* this is because magic_number is not available in be nets */

  if (remote_mode && (net_type == NET_TYPE_PT || net_type == NET_TYPE_KK)) {

    net_data.zmq_metadata.zmq_in_edge_places = metadata_ptr[4];
    net_data.zmq_metadata.zmq_out_edge_places = metadata_ptr[5];
    net_data.zmq_metadata.zmq_in_edge_startid = metadata_ptr[6];
    net_data.zmq_metadata.zmq_out_edge_startid = metadata_ptr[7];

    if (net_data.zmq_metadata.zmq_in_edge_places == METADATA_FIELD_UNSET
        || net_data.zmq_metadata.zmq_out_edge_places == METADATA_FIELD_UNSET
        || net_data.zmq_metadata.zmq_in_edge_startid == METADATA_FIELD_UNSET
        || net_data.zmq_metadata.zmq_out_edge_startid == METADATA_FIELD_UNSET) {
      printf("ERROR: Remote mode is enabled, but ZMQ metadata is not set!\n");
      printf("Please provide the ZMQ metadata in the net file!\n");
      return 1;
    } else {
      printf("Remote mode ZMQ metadata:\n");
      printf("  places in input-edge: %u\n", net_data.zmq_metadata
                                            .zmq_in_edge_places);
      printf("  places in output edge: %u\n", net_data.zmq_metadata
                                              .zmq_out_edge_places);
      printf("  startid in input-edge: %u\n", net_data.zmq_metadata
                                              .zmq_in_edge_startid);
      printf("  startid in output edge: %u\n", net_data.zmq_metadata
                                              .zmq_out_edge_startid);
    }
  } else if (remote_mode && (net_type == NET_TYPE_SPT 
                            || net_type == NET_TYPE_SKK
                            || net_type == NET_TYPE_SBE) ) {
    net_data.zmq_metadata.zmq_uplinks = metadata_ptr[4];
    net_data.zmq_metadata.zmq_downlinks = metadata_ptr[5];
    net_data.zmq_metadata.zmq_uplink_startid = metadata_ptr[6];
    net_data.zmq_metadata.zmq_downlink_startid = metadata_ptr[7];
    net_data.zmq_metadata.zmq_target_uplink_startid = metadata_ptr[8];

    if (net_data.zmq_metadata.zmq_uplinks == METADATA_FIELD_UNSET
        || net_data.zmq_metadata.zmq_downlinks == METADATA_FIELD_UNSET
        || net_data.zmq_metadata.zmq_uplink_startid == METADATA_FIELD_UNSET
        || net_data.zmq_metadata.zmq_downlink_startid == METADATA_FIELD_UNSET
        || net_data.zmq_metadata.zmq_target_uplink_startid 
           == METADATA_FIELD_UNSET) {
      printf("ERROR: Remote mode is enabled, but ZMQ metadata is not set!\n");
      printf("Please provide the ZMQ metadata in the net file!\n");
      return 1;
    } else {
      printf("Remote mode ZMQ metadata:\n");
      printf("  uplinks: %u\n", net_data.zmq_metadata.zmq_uplinks);
      printf("  downlinks: %u\n", net_data.zmq_metadata.zmq_downlinks);
      printf("  uplink_startid: %u\n", net_data.zmq_metadata
            .zmq_uplink_startid);
      printf("  downlink startid: %u\n", net_data.zmq_metadata
            .zmq_downlink_startid);
      printf("  target uplink startid: %u\n", net_data.zmq_metadata
            .zmq_target_uplink_startid);
    }
  }

  if (net_type == NET_TYPE_KK) net_data.glob_color_degree 
    = metadata_ptr[3];
  else net_data.glob_color_degree = 1;

  const uint32_t place_size = 
    net_data.glob_color_degree * (net_type == NET_TYPE_BE ? sizeof(be_place_t) 
                               : (net_type == NET_TYPE_SBE ? sizeof(be_place_t)
                               : (net_type == NET_TYPE_PT ? sizeof(pt_place_t) 
                               : (net_type == NET_TYPE_SPT ? sizeof(pt_place_t)
                               : sizeof(kk_place_subelem_t)))));
  uint32_t transitions_region_size = 0;
  if (net_type == NET_TYPE_BE || net_type == NET_TYPE_SBE) {
    transitions_region_size = net_data.transitions * 
                              (2 * net_data.transition_line_elems * 
                               sizeof(be_transition_elem_t));
  } else if (net_type == NET_TYPE_PT || net_type == NET_TYPE_SPT) {
    transitions_region_size = net_data.transitions * 
                              (2 * net_data.transition_line_elems * 
                               sizeof(pt_transition_elem_t));
  } else if (net_type == NET_TYPE_KK || net_type == NET_TYPE_SKK) {
    transitions_region_size = net_data.transitions * 
                              (2 * net_data.transition_line_elems * 
                               sizeof(kk_transition_subelem_t));
  }

  /* the following pointer points to where the marking array is in the alloca-*/
  /* ted data, which is the start of the data chunk + metadata                */
  net_data.net.dev_marking_arr = (void*) (dev_netarea_ptr + METADATA_PADDING);  
  
  printf("Net input is of type: ");
  if (net_type == NET_TYPE_BE) printf(NET_NAME_BE); 
  else if (net_type == NET_TYPE_PT) printf(NET_NAME_PT); 
  else if (net_type == NET_TYPE_KK) printf(NET_NAME_KK);
  else if (net_type == NET_TYPE_SPT) printf(NET_NAME_SPT);
  else if (net_type == NET_TYPE_SKK) printf(NET_NAME_SKK);
  else if (net_type == NET_TYPE_SBE) printf(NET_NAME_SBE);
  printf("\n");

  const uint32_t marking_area_bytes = ROUND_UP(net_data.places * place_size,
                                               PLACE_PADDING);
  const uint32_t pre_transition_area_bytes = 
    ROUND_UP(METADATA_PADDING + marking_area_bytes, TRANSITION_AREA_PADDING); 
  net_data.net.dev_transitions_arr = (void*) (dev_netarea_ptr 
                                              + pre_transition_area_bytes);

  printf("Places: %u, Transitions: %u, Indices in transition line: %u\n",
         net_data.places, net_data.transitions, net_data.transition_line_elems);

  /************************************************ net security assertions ***/
  ASSERT(net_data.transitions <= dev_props->max_concurrent_threads);

  /* padding check */
  if (transitions_region_size % TRANSITION_PADDING != 0) {
    printf("ERROR: Transitions region size (%u bytes) is unpadded, %u bytes!\n",
           transitions_region_size, TRANSITION_PADDING);
    return 1;
  }

  const uint32_t transition_area_512pad_ptr_begin_mod_TRANSITION_AREA_PADDING = 
    (uintptr_t) net_data.net.dev_transitions_arr % TRANSITION_AREA_PADDING;
  if (transition_area_512pad_ptr_begin_mod_TRANSITION_AREA_PADDING != 0) {
    printf("ERROR: Transitions region is not padded to %u bytes!\n",
           TRANSITION_AREA_PADDING);
    printf("Transition area padding: %u bytes\n",
           transition_area_512pad_ptr_begin_mod_TRANSITION_AREA_PADDING);
    return 1;
  }

  /********************* transition tune strategy setup with texture memory ***/
  fragm_transitions_tex_t t_tex = alloc_fragmented_transition_tex(
    dev_props, transitions_region_size, net_data.transitions,
    (uint8_t*) net_data.net.dev_transitions_arr, net_type
  );

  /********************************************************* resuming setup ***/
  const uint32_t placelock_amount = net_data.places;
  /* we need a lock for each place */
  uint32_t* dev_placelocks = 0;

  /* we do not have placelocks for KK/PT nets, therefore only for BE: */
  if (net_type == NET_TYPE_BE) {
    cudaMalloc(&dev_placelocks, placelock_amount * sizeof(uint32_t));
    cudaMemsetAsync(dev_placelocks, 0, placelock_amount * sizeof(uint32_t),
                    io_stream);
  }
  
  /* global steps tracker or control flag */
  int32_t* dev_run_cond;
  if (run_cond == RUN_COND_CTRLFLAG_IDENTIFIER) {
    cudaMallocManaged(&dev_run_cond, sizeof(int32_t));
    *dev_run_cond = SIMULATION_RUN;
    managed_ctrlflag_ptr = dev_run_cond;
    /* now, the ctrflag is retrievable from e.g. java context */
  } else { /* run_cond = step counter, completely on device for performance */
    cudaMalloc(&dev_run_cond, sizeof(int32_t));
    cudaMemcpyAsync(dev_run_cond, &run_cond, sizeof(int32_t), 
                    cudaMemcpyHostToDevice, io_stream);
  }

  /* make dev_props available to device for kernel exec*/
  device_prop_t* dev_dev_props;
  cudaMalloc(&dev_dev_props, sizeof(device_prop_t));
  cudaMemcpyAsync(dev_dev_props, dev_props, sizeof(device_prop_t), 
                  cudaMemcpyHostToDevice, io_stream);

  int32_t minGridSize;
  int32_t optimalBlockSize;
  cudaError_t err;
  if (net_type == NET_TYPE_BE) {
    err = cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &optimalBlockSize, transition_kernel<false, be_net_t>, 0, 0
    );
  } else if (net_type == NET_TYPE_SBE) {
    err = cudaOccupancyMaxPotentialBlockSize(
      /* since step version does not exist, false */
      &minGridSize, &optimalBlockSize, sync_transition_kernel<be_net_t>, 0, 0
    );
  } else if (net_type == NET_TYPE_PT) {
    err = cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &optimalBlockSize, transition_kernel<true, pt_net_t>, 0, 0
    );
  } else if (net_type == NET_TYPE_SPT) {
    err = cudaOccupancyMaxPotentialBlockSize(
      /* since step version does not exist, false */
      &minGridSize, &optimalBlockSize, sync_transition_kernel<pt_net_t>, 0, 0
    );
  } else if (net_type == NET_TYPE_KK) {
    err = cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &optimalBlockSize, transition_kernel<true, kk_net_t>, 0, 0
    );
  } else if (net_type == NET_TYPE_SKK) {
    err = cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &optimalBlockSize, sync_transition_kernel<kk_net_t>, 0, 0
    );
  }

  if (err != cudaSuccess) {
    fprintf(stderr, "Occupancy calculation failed: %s\n",
            cudaGetErrorString(err));
    exit(1);
  }

  uint32_t numBlocks = (net_data.transitions + optimalBlockSize - 1) 
                        / optimalBlockSize;
  printf("Optimal <<<%d, %d>>> launch for %d workers\n", numBlocks, 
         optimalBlockSize, net_data.transitions);
  printf("Total threads: %d (overshoot: %d)\n",
          numBlocks * optimalBlockSize,
          numBlocks * optimalBlockSize - net_data.transitions);

  /* synchronize streams before kernel launch to ensure data is ready */
  cudaStreamSynchronize(io_stream);

  /************************************************************* main phase ***/
  thread_t** workers = 0;
  uint32_t   num_workers = 0;
  uint32_t*  managed_uplink_msg_ptr = 0;   /* for synchronous transition nets */
  uint32_t*  managed_downlink_msg_ptr = 0;
  if (remote_mode && (net_type == NET_TYPE_PT || net_type == NET_TYPE_KK)) {
    uint32_t* managed_in_edge_ptr = (uint32_t*) (net_data.net.dev_marking_arr);
    uint32_t* managed_out_edge_ptr = (uint32_t*) 
      ((uint8_t*) net_data.net.dev_marking_arr + 
       net_data.places * net_data.glob_color_degree * place_size
       - net_data.zmq_metadata.zmq_out_edge_places * net_data.glob_color_degree 
                                                   * place_size);

    uint32_t elems_in_out_edge = net_data.zmq_metadata.zmq_out_edge_places 
                                 * net_data.glob_color_degree;

    num_workers = zmq_remote_place_init_threads(net_data.zmq_metadata,
                                                managed_in_edge_ptr,
                                                managed_out_edge_ptr,
                                                elems_in_out_edge,
                                                &workers);

  } else if (remote_mode && (net_type == NET_TYPE_SPT
                            || net_type == NET_TYPE_SKK
                            || net_type == NET_TYPE_SBE)) {
    /* allocate the message buffers, one for amount of uplinks, one for down. */
    if (net_data.zmq_metadata.zmq_uplinks > 0) {
      cudaMallocManaged(&managed_uplink_msg_ptr,
                        net_data.zmq_metadata.zmq_uplinks * sizeof(uint32_t));
      cudaMemsetAsync(managed_uplink_msg_ptr, 0xFF,
                      net_data.zmq_metadata.zmq_uplinks * sizeof(uint32_t),
                      io_stream);
    }
    
    if (net_data.zmq_metadata.zmq_downlinks > 0) {
      cudaMallocManaged(&managed_downlink_msg_ptr,
                        net_data.zmq_metadata.zmq_downlinks * sizeof(uint32_t));
      cudaMemsetAsync(managed_downlink_msg_ptr, 0xFF,
                      net_data.zmq_metadata.zmq_downlinks * sizeof(uint32_t),
                      io_stream);
    }

    cudaStreamSynchronize(io_stream);

    uint32_t num_uplink_downlink_threads = zmq_uplink_downlink_init_threads(
      managed_uplink_msg_ptr,
      managed_downlink_msg_ptr,
      net_data.zmq_metadata.zmq_uplinks,
      net_data.zmq_metadata.zmq_downlinks,
      net_data.zmq_metadata.zmq_uplink_startid,
      net_data.zmq_metadata.zmq_downlink_startid,
      &workers
    );
  }

  clock_gettime(CLOCK_TYPE, &prekernel);
  double setup_time = timespec_diff_ms(&start, &prekernel);
  printf("Setup time: %.9f ms\n", setup_time);

  /************************************************************ kernel exec ***/

  if (net_type == NET_TYPE_BE) {
    if (run_cond == RUN_COND_CTRLFLAG_IDENTIFIER) {
      transition_kernel<false, be_net_t><<<numBlocks, optimalBlockSize, 
                                           0, compute_stream>>>(
        dev_dev_props, to_net_data_T<be_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex
      );
    } else {
      transition_kernel<true, be_net_t><<<numBlocks, optimalBlockSize, 
                                          0, compute_stream>>>(
        dev_dev_props, to_net_data_T<be_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex
      );
    }
  }
  else if (net_type == NET_TYPE_PT) {
    if (run_cond == RUN_COND_CTRLFLAG_IDENTIFIER) {
      transition_kernel<false, pt_net_t><<<numBlocks, optimalBlockSize, 
                                           0, compute_stream>>>(
        dev_dev_props, to_net_data_T<pt_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex
      );
    } else {
      transition_kernel<true, pt_net_t><<<numBlocks, optimalBlockSize, 
                                          0, compute_stream>>>(
        dev_dev_props, to_net_data_T<pt_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex
      );
    }
  }
  else if (net_type == NET_TYPE_KK) {
    if (run_cond == RUN_COND_CTRLFLAG_IDENTIFIER) {
      transition_kernel<false, kk_net_t><<<numBlocks, optimalBlockSize, 
                                           0, compute_stream>>>(
        dev_dev_props, to_net_data_T<kk_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex
      );
    } else {
      transition_kernel<true, kk_net_t><<<numBlocks, optimalBlockSize, 
                                          0, compute_stream>>>(
        dev_dev_props, to_net_data_T<kk_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex
      );
    }
  } 
  else if (net_type == NET_TYPE_SPT) {
    if (run_cond == RUN_COND_CTRLFLAG_IDENTIFIER) {
      sync_transition_kernel<pt_net_t><<<numBlocks, optimalBlockSize, 
                               0, compute_stream>>>(
        dev_dev_props, to_net_data_T<pt_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex, managed_uplink_msg_ptr, managed_downlink_msg_ptr
      );
    } else {
      printf("ERROR: Step mode not supported for SPT nets!\n");
      return 1;
    }
  }
  else if (net_type == NET_TYPE_SKK) {
    if (run_cond == RUN_COND_CTRLFLAG_IDENTIFIER) {
      sync_transition_kernel<kk_net_t><<<numBlocks, optimalBlockSize, 
                               0, compute_stream>>>(
        dev_dev_props, to_net_data_T<kk_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex, managed_uplink_msg_ptr, managed_downlink_msg_ptr
      );
    } else {
      printf("ERROR: Step mode not supported for SKK nets!\n");
      return 1;
    }
  }
  else if (net_type == NET_TYPE_SBE) {
    if (run_cond == RUN_COND_CTRLFLAG_IDENTIFIER) {
      sync_transition_kernel<be_net_t><<<numBlocks, optimalBlockSize, 
                               0, compute_stream>>>(
        dev_dev_props, to_net_data_T<be_net_t>(net_data), dev_run_cond,
        dev_placelocks, t_tex, managed_uplink_msg_ptr, managed_downlink_msg_ptr
      );
    } else {
      printf("ERROR: Step mode not supported for SBE nets!\n");
      return 1;
    }
  }

  /* if remote_mode, this is done in start_zmq*/
  if (run_cond == RUN_COND_CTRLFLAG_IDENTIFIER) {
    char in;
    while (true) {
      in = getchar();
      if (in == 'T' || in == 't') {
        sim_stop();
        break;
      } else if (in == 'P' || in == 'p') {
        sim_togglepause();
      }
    }

    if (remote_mode) {
      zmq_stop_threads(workers, num_workers);
    }
  }
  
  cudaStreamSynchronize(compute_stream);
  CUDA_ERR_ASSERT();

  clock_gettime(CLOCK_TYPE, &postkernel);
  double kernel_time = timespec_diff_ms(&prekernel, &postkernel);
  printf("Kernel execution time: %.9f ms\n", kernel_time);

  /********************************************************** cleanup phase ***/
  destroy_fragmented_transition_tex(t_tex);
  SAFE_CUDAFREE(dev_dev_props);
  if (net_type == NET_TYPE_BE) SAFE_CUDAFREE(dev_placelocks);
  SAFE_CUDAFREE(dev_run_cond);

  /* async'd stream back to host if we were on device memory */
  if (work_on_dev_mem) {
    uint8_t* marking_bytes_host = ((uint8_t*) managed_filedata_ptr) 
                                  + METADATA_PADDING;
    const uint32_t marking_bytes = net_data.places * place_size;
    cudaMemcpyAsync(marking_bytes_host, net_data.net.dev_marking_arr,
                    marking_bytes, cudaMemcpyDeviceToHost,
                    io_stream);
    cudaStreamSynchronize(io_stream); /* wait for copy completion */
    SAFE_CUDAFREE(dev_netarea_ptr);
    
  }
  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(io_stream);

  if (marking_out_file != 0) {
    uint32_t marking_bytes = net_data.places * place_size;
    uint8_t* marking_bytes_from_out = ((uint8_t*) managed_filedata_ptr) 
                                      + METADATA_PADDING;
    uint32_t bytes_written = fwrite(marking_bytes_from_out, 1, marking_bytes, 
                                    marking_out_file);
  }
  return 0;
}

#if !defined(COMPILED_AS_LIBRARY)
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
 *   ./executable <steps/0> <file-in> <file/pipe-out> [--remote/-r 
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

  printf("KK/PT/BE-Net-Simulation Core via CUDA\n");

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

  if (argc > 3 && (strcmp(argv[argc-3], "--remote") == 0 
                 || strcmp(argv[argc-3], "-r") == 0)) {
    remote_mode = true;
    if (strcmp(argv[argc-2], NO_REMOTE_ADDR) != 0) {
      recv_port = argv[argc-2];
    }
    if (strcmp(argv[argc-1], NO_REMOTE_ADDR) != 0) {
      send_addr = argv[argc-1];
    }

    if (recv_port == 0 && send_addr == 0) {
      fprintf(stderr, "Error: Remote mode requires a receive port and a "
                      "send address!\n");
      return 1;
    }
  }

  int32_t steps = atoi(argv[1]);
  if (steps == RUN_COND_CTRLFLAG_IDENTIFIER) {
    printf("No steps, use T and P key for control\n");
  } else {

    if (remote_mode) {
      printf("remote mode with steps is not supported!\n");
      return 1;
    }

    printf("Running simulation for %d steps\n", steps);
  }

  uint8_t* host_managed_data; /* bytes */
  uint32_t data_bytes;
  char* fileout;

  if (argc == 3) {
    data_bytes = fl_stdin_managed_malloc(&host_managed_data);
    fileout = argv[2];
  } else {
    data_bytes = fl_file_managed_malloc(argv[2], &host_managed_data);
    fileout = argv[3];
  }
  
  FILE *f = fopen(fileout, "wb");
  if (!f) {
    fprintf(stderr, "Error: Could not open output file '%s'.\n", fileout);
    return 1;
  }
  
  /* litle prelook to force dev mem on remote mode */
  bool forcedev = ((uint32_t*) host_managed_data)[MAGIC_NUMBER_OFFSET] 
                  == NET_TYPE_SPT
                  || ((uint32_t*) host_managed_data)[MAGIC_NUMBER_OFFSET] 
                     == NET_TYPE_SKK
                  || ((uint32_t*) host_managed_data)[MAGIC_NUMBER_OFFSET] 
                     == NET_TYPE_SBE;

  dissect_netdata_run_kernel(
    &dev_props, host_managed_data, data_bytes, !remote_mode || forcedev, steps,f
  );

  ASSERT(fclose(f) == 0);
  SAFE_CUDAFREE(host_managed_data);
  return 0;
}
#endif

#if defined(JAVA_SUPPORT)
typedef jlong j_ptr_t;

/* global pointer for JNI functions to the memory the kernel will act on */
j_ptr_t java_host_managed_data;

#define JAVAFUNC_PREFIX CONCATENATE(Java_, JNI_HEADER_NAME)

#define JAVA_X_CREATE_GPU_VISIBLE_POINTER \
  CONCATENATE(JAVAFUNC_PREFIX, _createGPUVisiblePointer)
#define JAVA_X_DESTROY_GPU_VISIBLE_POINTER \
  CONCATENATE(JAVAFUNC_PREFIX, _destroyGPUVisiblePointer)
#define JAVA_X_SIMULATE_ON_GPU \
  CONCATENATE(JAVAFUNC_PREFIX, _simulateOnGpu)
#define JAVA_X_SIMULATE_DISTRIBUTED_ON_GPU \
  CONCATENATE(JAVAFUNC_PREFIX, _simulateDistributedOnGpu)
#define JAVA_X_PAUSE_SIMULATION \
  CONCATENATE(JAVAFUNC_PREFIX, _pauseSimulation)
#define JAVA_X_STOP_SIMULATION \
  CONCATENATE(JAVAFUNC_PREFIX, _stopSimulation)
/**
 * JAVA_X_CREATE_GPU_VISIBLE_POINTER - creates a GPU-visible pointer
 * @env: JNI environment pointer, managed by JNI
 * @clazz: JNI class pointer, managed by JNI since this is a static function
 * @original_ptr: first arg in Java method, pointer to the original net data,
 *                which must have the same layout as defined above for net-data
 * @len: second arg in Java method, byte amount of the net-data where the ptr
 *       above points to
 *
 * This function creates a GPU-visible pointer by allocating managed memory.
 * It is called by Java inside the class GPUConnector with the method:
 * 
 * - public static long createGPUVisiblePointer(long original_ptr, int len)
 * 
 * Also, this and the other JNIEXPORT functions follow the naming structure
 * of Java_<JNI_HEADER_NAME>_<function_name> to ensure uniqueness.
 * JNI_HEADER_NAME is the name of the JNI header file without the .h and
 * contains the fully qualified class name, e.g., com_example_GPUConnector
 * 
 * Context: JNI handles the full context, the function is unused within here,
 *          JAVA_X_DESTROY_GPU_VISIBLE_POINTER must be called after use!
 */
JNIEXPORT j_ptr_t JNICALL JAVA_X_CREATE_GPU_VISIBLE_POINTER(
  JNIEnv *env __attribute__((unused)),
  jclass clazz __attribute__((unused)),
  j_ptr_t original_ptr,
  jint len
) {
  uint8_t* javas_data_ptr = (uint8_t*) original_ptr;
  uint8_t* new_byte_ptr;
  cudaMallocManaged((void**)&new_byte_ptr, len);
  cudaMemcpy(new_byte_ptr, javas_data_ptr, len, cudaMemcpyHostToDevice);
  /* store the pointer in a global variable for later use */
  java_host_managed_data = (j_ptr_t) new_byte_ptr;
  return (j_ptr_t) new_byte_ptr;
}

/**
 * JAVA_X_DESTROY_GPU_VISIBLE_POINTER - destroys a GPU-visible pointer
 * @env: JNI environment pointer, managed by JNI
 * @clazz: JNI class pointer, managed by JNI since this is a static function
 * @ptr: first arg in Java method, pointer to the GPU-visible data to be freed
 *
 * This function destroys a GPU-visible pointer by freeing the managed memory.
 * It is called by Java inside the class GPUConnector with the method:
 *
 * - public static boolean destroyGPUVisiblePointer(long ptr)
 *
 * Context: JNI handles the full context, the function is unused within here,
 *          JAVA_X_CREATE_GPU_VISIBLE_POINTER must be called before use!
 */
JNIEXPORT jboolean JNICALL JAVA_X_DESTROY_GPU_VISIBLE_POINTER(
  JNIEnv *env __attribute__((unused)),
  jclass clazz __attribute__((unused)),
  j_ptr_t ptr
) {
  if (ptr == 0) return 0;
  uint8_t* javas_data_ptr = (uint8_t*) ptr;
  if (javas_data_ptr == NULL) return 0;
  cudaFree(javas_data_ptr);
  java_host_managed_data = 0; /* reset the global pointer */
  return 0;
}

/**
 * JAVA_X_SIMULATE_ON_GPU - simulates a petri net on the GPU
 * @env: JNI environment pointer, managed by JNI
 * @clazz: JNI class pointer, managed by JNI since this is a static function
 * @gpu_visible_ptr: first arg in Java method, pointer to the GPU-visible data
 *                which must have the same layout as defined above for net-data
 * @len: second arg in Java method, byte amount of the net-data where the ptr
 *       above points to
 * @steps: third arg in Java method, number of steps to simulate
 * 
 * This function runs the petri net simulation on the GPU using the
 * dissect_netdata_run_kernel function. It is called by Java inside the class
 * GPUConnector with the method:
 * 
 * - public static void simulateOnGpu(long gpu_visible_ptr, int len, int steps)
 * 
 * Context: JNI handles the full context, the function is unused within here,
 *          JAVA_X_CREATE_GPU_VISIBLE_POINTER must be called before use!
 * 
 * 
 */
JNIEXPORT void JNICALL JAVA_X_SIMULATE_ON_GPU(
  JNIEnv *env __attribute__((unused)),
  jclass clazz __attribute__((unused)),
  j_ptr_t gpu_visible_ptr, /* managed */
  jint len,
  jint steps
) {
  clock_gettime(CLOCK_TYPE, &start);
  device_prop_t dev_props = get_device_properties();
  uint8_t* host_managed_ptr = (uint8_t*) gpu_visible_ptr;
  dissect_netdata_run_kernel(
    &dev_props, host_managed_ptr, len, false, steps, 0
  );                                  /* |-> for unified mem  */
}

/**
 * JAVA_X_SIMULATE_DISTRIBUTED_ON_GPU - simulates a distributed petri net on the GPU
 * @env: JNI environment pointer, managed by JNI
 * @clazz: JNI class pointer, managed by JNI since this is a static function
 * @gpu_visible_ptr: first arg in Java method, pointer to the GPU-visible data
 *                which must have the same layout as defined above for net-data
 * @len: second arg in Java method, byte amount of the net-data where the ptr
 *       above points to
 * @steps: third arg in Java method, number of steps to simulate
 * @recv_port_str: fourth arg in Java method, string for the receive port
 * @send_addr_str: fifth arg in Java method, string for the send address
 *
 * This function runs the distributed petri net simulation on the GPU using the
 * dissect_netdata_run_kernel function. It is called by Java inside the class
 * GPUConnector with the method:
 * - public static void simulateDistributedOnGpu(long gpu_visible_ptr, 
 *                                               int len, int steps,
 *                                               String recv_port,
 *                                               String send_addr)
 * Context: JNI handles the full context, the function is unused within here,
 *          JAVA_X_CREATE_GPU_VISIBLE_POINTER must be called before use!
 *          Remote mode is enabled in this function.
 * Return: void 
 *
 */
JNIEXPORT void JNICALL JAVA_X_SIMULATE_DISTRIBUTED_ON_GPU(
  JNIEnv *env __attribute__((unused)),
  jclass clazz __attribute__((unused)),
  j_ptr_t gpu_visible_ptr, /* managed */
  jint len,
  jint steps,
  jstring recv_port_str,
  jstring send_addr_str
) {
  const char *recv_port_cstr = env->GetStringUTFChars(recv_port_str, 0);
  const char *send_addr_cstr = env->GetStringUTFChars(send_addr_str, 0);
  printf("Using ZMQ recv port: %s\n", recv_port_cstr);
  printf("Using ZMQ send address: %s\n", send_addr_cstr);
  recv_port = strdup(recv_port_cstr);
  send_addr = strdup(send_addr_cstr);
  remote_mode = true;
  clock_gettime(CLOCK_TYPE, &start);
  device_prop_t dev_props = get_device_properties();
  uint8_t* host_managed_ptr = (uint8_t*) gpu_visible_ptr;
  dissect_netdata_run_kernel(
    &dev_props, host_managed_ptr, len, false, steps, 0
  );
  env->ReleaseStringUTFChars(recv_port_str, recv_port_cstr);
  env->ReleaseStringUTFChars(send_addr_str, send_addr_cstr);
  if (recv_port) { free(recv_port); recv_port = nullptr; }
  if (send_addr) { free(send_addr); send_addr = nullptr; }
}

/**
 * JAVA_X_PAUSE_SIMULATION - pauses the simulation
 * 
 * This function toggles the pause state of the simulation. It is called by
 * Java inside the class GPUConnector with the method:
 * 
 * - public static void pauseSimulation()
 * 
 * Context: JNI handles the full context, and a simulation must be started w/o
 *          a step counter. Otherwise this won't work.
 * 
 * Return: void
 */
JNIEXPORT void JNICALL JAVA_X_PAUSE_SIMULATION(
  JNIEnv *env __attribute__((unused)),
  jclass clazz __attribute__((unused))
) {
  if (managed_ctrlflag_ptr) {
    sim_togglepause();
  } else {
    printf("No simulation running, cannot pause...\n");
  }
}

/**
 * JAVA_X_STOP_SIMULATION - stops the simulation
 * 
 * This function stops the simulation by setting the control flag to terminate.
 * It is called by Java inside the class GPUConnector with the method:
 * 
 * - public static void stopSimulation()
 * 
 * Context: JNI handles the full context, and a simulation must be started w/o
 *          a step counter. Otherwise this won't work.
 * 
 * Return: void
 */
JNIEXPORT void JNICALL JAVA_X_STOP_SIMULATION(
  JNIEnv *env __attribute__((unused)),
  jclass clazz __attribute__((unused))
) {
  if (managed_ctrlflag_ptr) {
    sim_stop();
  } else {
    printf("No simulation running, cannot stop...\n");
  }
}
#endif

#pragma PREPROCESSOR_MARKER_END