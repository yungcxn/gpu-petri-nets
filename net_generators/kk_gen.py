#!/usr/bin/env python3
"""
fixed petri net generator - generates different net types with proper logic
usage: python kk_gen.py <arg> <mode> <colors>
"""

import sys
import numpy as np
import random

OPT_PADDING = 128
OPT_TRANS_PAD = int(OPT_PADDING / 4)
OPT_TRANS_PADDING = 512
MAGIC_NUMBER = 2
SIGNED_PLACES = True
UINT32SIZE = 4 # byte

def pack_int32(x):
  return x.to_bytes(4, byteorder=sys.byteorder, signed=SIGNED_PLACES)

def pack_uint32(x):
  return x.to_bytes(4, byteorder=sys.byteorder, signed=False)

def round_up(x, mult):
  return ((x + mult - 1) // mult) * mult if mult > 0 else x

def make_marking(places, glob_color_degree, marking_arrs):
  mark = np.zeros(places * glob_color_degree, dtype=np.int32)
  for elem in marking_arrs:
    placeref = elem[0]
    color_arr = elem[1:]
    for i, wci in enumerate(color_arr):
      mark[placeref * glob_color_degree + i] = wci
  return mark.tobytes()

def make_output(places, transitions, padded_line_uint32s, glob_color_degree, marking, trans_arrays):
  marking_bytes = len(marking)
  
  # Metadata area: exactly 128 bytes
  # First 4 uint32_t values, then padding, then magic number at offset 124
  metadata_padding = 128 - 5 * 4  # 108 bytes padding
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
    pack_uint32(places),              # offset 0
    pack_uint32(transitions),         # offset 4  
    pack_uint32(padded_line_uint32s), # offset 8
    pack_uint32(glob_color_degree),   # offset 12
    pad_bytes1,                       # padding to offset 124
    pack_uint32(MAGIC_NUMBER),        # offset 124 (31*4)
    marking,                          # marking array
    pad_bytes2,                       # marking padding
    pre_trans_pad_bytes               # pre-transition padding
  ] + trans_arrays + [trans_pad_bytes]
  
  return b''.join(parts)

def make_colored_transition(padded_line_uint32s, in_places, out_places, glob_color_degree):
  trans_in = bytearray([0xFF] * padded_line_uint32s * UINT32SIZE)
  trans_out = bytearray([0xFF] * padded_line_uint32s * UINT32SIZE)
  for i, pref_color_arr_i in enumerate(in_places):
    glob_i = i*(glob_color_degree+1)
    p = pref_color_arr_i[0]
    trans_in[glob_i*UINT32SIZE:glob_i*UINT32SIZE+UINT32SIZE] = pack_int32(p)
    colors = pref_color_arr_i[1:]
    if len(colors) != glob_color_degree:
      print("Misaligned transition lists")
      sys.exit(1)
    for j, weight_for_cj in enumerate(colors):
      glob_j = glob_i + j + 1
      trans_in[glob_j*UINT32SIZE:glob_j*UINT32SIZE+UINT32SIZE] = pack_int32(weight_for_cj)

  for i, pref_color_arr_i in enumerate(out_places):
    glob_i = i*(glob_color_degree+1)
    p = pref_color_arr_i[0]
    trans_out[glob_i*UINT32SIZE:glob_i*UINT32SIZE+UINT32SIZE] = pack_int32(p)
    colors = pref_color_arr_i[1:]
    if len(colors) != glob_color_degree:
      print("Misaligned transition lists")
      sys.exit(1)
    for j, weight_for_cj in enumerate(colors):
      glob_j = glob_i + j + 1
      trans_out[glob_j*UINT32SIZE:glob_j*UINT32SIZE+UINT32SIZE] = pack_int32(weight_for_cj)

  return [trans_in, trans_out]

def gen_relay_race(elements, colors):
  places = elements * 2
  initial = []
  initial_colors = list(range(1, colors+1))
  initial = [[0] + initial_colors]
  marking = make_marking(places, colors, initial)
  trans_arrays = []
  max_trans_elems_per_line = 2
  unpadded_line_uint32s = max_trans_elems_per_line * (colors + 1)
  padded_line_uint32s = round_up(unpadded_line_uint32s, OPT_TRANS_PAD) 
  for i in range(elements * 2):
    in_p = [[i] + [i + j for j in initial_colors]]
    out_p = [[i + 1] + [i + j + 1 for j in initial_colors]] if i + 1 < elements*2 else [] 
    trans_arrays.extend(make_colored_transition(padded_line_uint32s, in_p, out_p, colors))
  return make_output(places, elements * 2, padded_line_uint32s, colors, marking, trans_arrays)

def gen_min_cycles(elements, colors):
  places = elements * 3
  transitions = elements * 2
  # token at start of each element (places 0, 3, 6, ...)
  initial_colors = list(range(1, colors+1))
  initial = [[3*i] + initial_colors for i in range(elements)]
  marking = make_marking(places, colors, initial)
  trans_arrays = []
  max_trans_elems_per_line = 2
  unpadded_line_uint32s = max_trans_elems_per_line * (colors + 1)
  padded_line_uint32s = round_up(unpadded_line_uint32s, OPT_TRANS_PAD) 
  for e in range(elements):
    start, left, right = 3 * e, 3 * e + 1, 3 * e + 2
    A = [[start] + initial_colors]
    B = [[left] + initial_colors, [right] + initial_colors]
    trans_arrays.extend(make_colored_transition(padded_line_uint32s, A, B, colors))
    trans_arrays.extend(make_colored_transition(padded_line_uint32s, B, A, colors))
  
  return make_output(places, transitions, padded_line_uint32s, colors, marking, trans_arrays)

def gen_philosophers(n, colors): # only one is used! grows the place vec
  if n == 0:
    raise ValueError("number of philosophers must be > 0")
  places = n * 3
  transitions = n * 2
  # each philosopher i has: fork_i, thinking_i, eating_i
  # initial: all forks available, all thinking
  initial = []
  for i in range(n):
    F = [3 * i, 1] + ([0] * (colors -1)) 
    T = [3 * i + 1, 1] + ([0] * (colors -1)) 
    initial.extend([F, T])  # fork and thinking
  marking = make_marking(places, colors, initial)
  trans_arrays = []
  max_trans_elems_per_line = 4
  unpadded_line_uint32s = max_trans_elems_per_line * (colors + 1)
  padded_line_uint32s = round_up(unpadded_line_uint32s, OPT_TRANS_PAD) 
  for i in range(n):
    fork_left = 3 * i
    thinking = 3 * i + 1  
    eating = 3 * i + 2
    fork_right = 3 * ((i + 1) % n)  # circular
    trans_arrays.extend(make_colored_transition(padded_line_uint32s, [[x, 1] + ([0] * (colors -1)) for x in sorted([fork_left, thinking, fork_right])], [[eating, 1] + ([0] * (colors -1))], colors))
    trans_arrays.extend(make_colored_transition(padded_line_uint32s, [[eating, 1] + ([0] * (colors -1)) ], [[x, 1] + ([0] * (colors -1))  for x in sorted([fork_left, thinking, fork_right])], colors))
  return make_output(places, transitions, padded_line_uint32s, colors, marking, trans_arrays)

def gen_conflict_places(n, colors):
  places = n * 2
  transitions = n * 2
  initial = []
  for x in range(n):
    initial_colors = [random.randint(2, 100_000) for _ in range(colors)]
    initial.append([x] + initial_colors)
  marking = make_marking(places, colors, initial)
  trans_arrays = []
  max_trans_elems_per_line = n
  unpadded_line_uint32s = max_trans_elems_per_line * (colors + 1)
  padded_line_uint32s = round_up(unpadded_line_uint32s, OPT_TRANS_PAD) 
  out_places_list = [[n + i] + [1]*colors for i in range(n)]
  for i in range(n):
    trans_arrays.extend(make_colored_transition(
      padded_line_uint32s, initial, [out_places_list[i]], colors))
  for i in range(n):
    trans_arrays.extend(make_colored_transition(
      padded_line_uint32s, [out_places_list[i]], initial, colors))
  return make_output(places, transitions, padded_line_uint32s, colors, marking, trans_arrays)

# main execution
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("usage: python netgen.py <net_type> <arg>")
    print("net_type: 0=relay, 3=min_cycles, 4=philosophers, 6=conflict")
    sys.exit(1)

  net_type = int(sys.argv[1])
  arg = int(sys.argv[2])
  colors = int(sys.argv[3])
  
  if arg == 0 or colors == 0:
    print("arg&colors must be > 0")
    sys.exit(1)

  try:
    if net_type == 0:
      data = gen_relay_race(arg, colors)
    elif net_type == 3:
      data = gen_min_cycles(arg, colors)
    elif net_type == 4:
      data = gen_philosophers(arg, colors)
    elif net_type == 6:
      data = gen_conflict_places(arg, colors)
    else:
      print(f"unknown net type: {net_type}")
      sys.exit(1)
    
    sys.stdout.buffer.write(data)
    
  except Exception as e:
    print(f"error: {e}", file=sys.stderr)
    sys.exit(1)