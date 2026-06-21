#!/usr/bin/env python3
"""
fixed petri net generator - generates different net types with proper logic
usage: python pt_gen.py <arg> <mode>
"""

import sys
import numpy as np
import random

OPT_PADDING = 128
OPT_TRANS_PAD = int(OPT_PADDING / 8)
OPT_TRANS_PADDING = 512
MAGIC_NUMBER = 1
SIGNED_PLACES = True

def pack_int32(x):
  return x.to_bytes(4, byteorder=sys.byteorder, signed=SIGNED_PLACES)

def pack_uint32(x):
  return x.to_bytes(4, byteorder=sys.byteorder, signed=False)

def round_up(x, mult):
  return ((x + mult - 1) // mult) * mult if mult > 0 else x

def make_marking(size, marking_tuples):
  # PT nets use signed int32s according to C++ code comments
  mark = np.zeros(size, dtype=np.int32)
  for marking, placeref in marking_tuples:
    if 0 <= placeref < size:
      mark[placeref] = marking
  return mark.tobytes()

def make_output(places, transitions, trans_elems, marking, trans_arrays, in_edge, out_edge, in_edge_startid, out_edge_startid):
  marking_bytes = len(marking)
  
  # Metadata area: exactly 128 bytes
  # First 3 uint32_t values, then padding, then magic number at offset 124
  metadata_padding = 128 - 4 * 9  # 112 bytes padding (3 values + magic = 4 total)
  pad_bytes1 = bytearray([0xFF] * metadata_padding)
  
  # Marking area padding to 128-byte boundary
  padding2 = (OPT_PADDING - (marking_bytes % OPT_PADDING)) % OPT_PADDING
  pad_bytes2 = bytearray([0xFF] * padding2) if padding2 > 0 else bytearray()
  
  # Pre-transition area padding to 512-byte boundary
  pre_trans_bytes = 128 + marking_bytes + padding2  # metadata + marking + marking padding
  pre_trans_padding = (OPT_TRANS_PADDING - (pre_trans_bytes % OPT_TRANS_PADDING)) % OPT_TRANS_PADDING
  pre_trans_pad_bytes = bytearray([0xFF] * pre_trans_padding)
  
  # Transition area padding to 512-byte boundary
  trans_bytes = sum(len(arr) for arr in trans_arrays)
  trans_padding = (OPT_TRANS_PADDING - (trans_bytes % OPT_TRANS_PADDING)) % OPT_TRANS_PADDING
  trans_pad_bytes = bytearray([0xFF] * trans_padding)
  
  # Construct the final binary with correct metadata layout
  parts = [
    pack_uint32(places),           # offset 0
    pack_uint32(transitions),      # offset 4  
    pack_uint32(trans_elems),      # offset 8
    pack_uint32(0xFFFFFFFF),       # offset 12
    pack_uint32(in_edge),          # offset 16
    pack_uint32(out_edge),         # offset 20
    pack_uint32(in_edge_startid),  # offset 24
    pack_uint32(out_edge_startid), # offset 28
    pad_bytes1,                    # padding to offset 124
    pack_uint32(MAGIC_NUMBER),     # offset 124 (31*4)
    marking,                       # marking array (signed int32s)
    pad_bytes2,                    # marking padding
    pre_trans_pad_bytes           # pre-transition padding
  ] + trans_arrays + [trans_pad_bytes]
  return b''.join(parts)

def make_transition_opt_place_indexed(size, in_places, out_places):
  trans_in = bytearray([0xFF] * (8 * size))
  trans_out = bytearray([0xFF] * (8 * size))
  for i, x in enumerate(in_places):
    w, p = x
    trans_in[i*8:i*8+4] = pack_int32(w)
    trans_in[i*8+4:i*8+8] = pack_int32(p)
  for i, x in enumerate(out_places):
    w, p = x
    trans_out[i*8:i*8+4] = pack_int32(w)
    trans_out[i*8+4:i*8+8] = pack_int32(p)
  return [trans_in, trans_out]

def gen_relay_race(elements, single_token=True, is_first_part=True):
  places = elements * 2
  if single_token:
    initial = [(1,0)] if is_first_part else []
  else:
    initial = [(1,i) for i in range(0, elements * 2, 2)]

  marking = make_marking(places, initial)
  
  trans_arrays = []
  real_elems = 2
  padded_real_elems = round_up(real_elems, OPT_TRANS_PAD) 
  for i in range(elements * 2 - 1):
    in_p = [(1, i)] 
    out_p = [(1, i + 1)]
    trans_arrays.extend(make_transition_opt_place_indexed(padded_real_elems, in_p, out_p))
  
  if is_first_part:
    return make_output(places, elements * 2, padded_real_elems, marking, trans_arrays, 1, 1, 100, 1000000000)
  else:
    return make_output(places, elements * 2, padded_real_elems, marking, trans_arrays, 1, 1, 1000000000, 100)


def gen_conflict_places(n, is_first_part=True):
  places = n * 2
  transitions = n
  
  # initial tokens in first n places (conflict set)
  initial = [(i+1, i) for i in range(n)]
  
  marking = make_marking(places, initial) if is_first_part else make_marking(places, [])
  trans_arrays = []
  real_elems = n
  padded_real_elems = round_up(real_elems, OPT_TRANS_PAD)
  if is_first_part:
    for i in range(n):
      trans_arrays.extend(make_transition_opt_place_indexed(
        padded_real_elems, initial, [(1, n + i)]))
  else:
    for i in range(n):
      trans_arrays.extend(make_transition_opt_place_indexed(
        padded_real_elems, [(1, i)], [(a,b+n) for a,b in initial]))
  if is_first_part:
    return make_output(places, transitions, padded_real_elems, marking, trans_arrays, n, n, 0, n)
  else:
    return make_output(places, transitions, padded_real_elems, marking, trans_arrays, n, n, n, 0)

# main execution
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("usage: python netgen.py <net_type> <arg> <part-of-2>")
    print("net_type: 0/1=relay, 2=binary_tree, 3=min_cycles, 4=philosophers, 6=conflict, 7=generator")
    sys.exit(1)

  net_type = int(sys.argv[1])
  arg = int(sys.argv[2])
  first_part = int(sys.argv[3]) == 1 if len(sys.argv) > 3 else exit(0) + 1
  
  
  if arg == 0:
    print("arg must be > 0")
    sys.exit(1)

  try:
    if net_type == 0:
      data = gen_relay_race(arg, single_token=True, is_first_part=first_part)
    elif net_type == 1:
      data = gen_relay_race(arg, single_token=False, is_first_part=first_part)
    elif net_type == 6:
      data = gen_conflict_places(arg, is_first_part=first_part)
    else:
      print(f"unknown net type: {net_type}")
      sys.exit(1)
    
    sys.stdout.buffer.write(data)
    
  except Exception as e:
    print(f"error: {e}", file=sys.stderr)
    sys.exit(1)