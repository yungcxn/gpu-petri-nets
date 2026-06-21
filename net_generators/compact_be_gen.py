#!/usr/bin/env python3
"""
fixed petri net generator - generates different net types with proper logic
usage: python be_gen.py <net_type> <arg> <mode> [pad]
mode: 0=place-indexed, 1=bytemasked, 2=bitmasked, 3=byte4masked 4=opt-place-indexed
"""

import sys
import numpy as np

OPT_PADDING = 128
OPT_TRANS_PAD = int(OPT_PADDING / 4)
OPT_TRANS_PADDING = 512
ORDER = False
MAGICNUMBER = 3

def pack_int32(x):
  return x.to_bytes(4, byteorder=sys.byteorder, signed=False)

def round_up(x, mult):
  return ((x + mult - 1) // mult) * mult if mult > 0 else x

def pack_bits(bits):
  bits = np.asarray(bits, dtype=np.uint8).flatten()
  if len(bits) % 8 != 0:
    raise ValueError(f"bit count {len(bits)} not divisible by 8")
  return np.packbits(bits).tobytes()

def make_output(places, transitions, marking, trans_arrays, trans_elems):
  marking_bytes = len(marking)
  
  # Metadata area: exactly 128 bytes
  # First 3 uint32_t values, then padding (no magic number for BE nets)
  # The C++ code will read magic_number from offset 124, but it won't be NET_TYPE_PT or NET_TYPE_KK
  # so it will default to BE net type
  metadata_padding = 128 - 4 * 4  # 116 bytes padding
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
    pack_int32(places),           # offset 0
    pack_int32(transitions),      # offset 4  
    pack_int32(trans_elems),      # offset 8
    pad_bytes1,                   # padding to fill 128 bytes total
    pack_int32(MAGICNUMBER),      # offset 12, magic number for BE nets
    marking,                      # marking array
    pad_bytes2,                   # marking padding
    pre_trans_pad_bytes          # pre-transition padding
  ] + trans_arrays + [trans_pad_bytes]
    
  return b''.join(parts)

def make_marking(size, marked_indices):
  mark = np.zeros(size, dtype=np.uint8)
  for i in marked_indices:
    if 0 <= i < size:
      mark[i] = 0xFF

  return mark.tobytes()


def make_transition_opt_place_indexed_mintrans(size, in_places, out_places):
  order = bytearray([0xFF] * (4 * size))

  ordered_places = sorted(set(in_places + out_places))
  out_places_set = set(out_places)

  for i, p in enumerate(ordered_places):
    if p in out_places_set:
      p |= 0x80000000 
    order[i*4:(i+1)*4] = pack_int32(p)

  return [order]


def gen_relay_race(elements, single_token=True, pad=0):
  places = round_up(elements * 2, pad) if pad else elements * 2
  
  # initial marking: token(s) at start
  if single_token:
    initial = [0]  # single token at p0
  else:
    initial = list(range(0, elements * 2, 2))  # tokens at p0, p2, p4, ...
 
  marking = make_marking(places, initial)
  
  trans_arrays = []
  real_elems = 2
  padded_real_elems = round_up(real_elems, OPT_TRANS_PAD) 
  # Create elements*2 transitions for full relay chain
  for i in range(elements * 2):
    in_p = [i]  # consume from current place
    out_p = [i + 1] if i + 1 < elements*2 else []  # produce to next place (if exists)
    trans_arrays.extend(make_transition_opt_place_indexed_mintrans(padded_real_elems, in_p, out_p))
  
  return make_output(places, elements * 2, marking, trans_arrays, padded_real_elems)

def gen_binary_tree(layers, pad=0):
  if layers <= 1:
    raise ValueError("layers must be > 1")
  
  places = round_up((2**layers) - 1, pad) if pad else (2**layers) - 1
  transitions = places - (2**(layers-1))  # internal nodes only
  
  # token at root (place 0)
  mode = "byte4"
  marking = make_marking(places, [0])
  
  trans_arrays = []
  real_elems = 3
  padded_real_elems = round_up(real_elems, OPT_TRANS_PAD) 
  for t in range(transitions):
    parent = t  # 0-based
    left_child = 2 * t + 1
    right_child = 2 * t + 2
    in_p = [parent]
    out_p = [c for c in [left_child, right_child] if c < places]
    trans_arrays.extend(make_transition_opt_place_indexed_mintrans(padded_real_elems, in_p, out_p))
  
  return make_output(places, transitions, marking, trans_arrays, padded_real_elems)

def gen_min_cycles(elements, pad=0):
  places = round_up(elements * 3, pad) if pad else elements * 3
  transitions = elements * 2
  
  # token at start of each element (places 0, 3, 6, ...)
  initial = [3 * i for i in range(elements)]
  mode = "byte4"
  marking = make_marking(places, initial)
  
  trans_arrays = []
  real_elems = 3
  padded_real_elems = round_up(real_elems, OPT_TRANS_PAD) 
  for e in range(elements):
    start, left, right = 3 * e, 3 * e + 1, 3 * e + 2
    trans_arrays.extend(make_transition_opt_place_indexed_mintrans(padded_real_elems, [start], [left, right]))
    trans_arrays.extend(make_transition_opt_place_indexed_mintrans(padded_real_elems, [left, right], [start]))
  
  return make_output(places, transitions, marking, trans_arrays, padded_real_elems)

def gen_philosophers(n, pad=0):
  if n == 0:
    raise ValueError("number of philosophers must be > 0")
  
  places = round_up(n * 3, pad) if pad else n * 3
  transitions = n * 2
  
  # each philosopher i has: fork_i, thinking_i, eating_i
  # initial: all forks available, all thinking
  initial = []
  for i in range(n):
    initial.extend([3 * i, 3 * i + 1])  # fork and thinking
  
  mode = "byte4"
  marking = make_marking(places, initial)
  
  trans_arrays = []
  real_elems = 4
  padded_real_elems = round_up(real_elems, OPT_TRANS_PAD) 
  for i in range(n):
    fork_left = 3 * i
    thinking = 3 * i + 1  
    eating = 3 * i + 2
    fork_right = 3 * ((i + 1) % n)  # circular
    trans_arrays.extend(make_transition_opt_place_indexed_mintrans(padded_real_elems, sorted([fork_left, thinking, fork_right]), [eating]))
    trans_arrays.extend(make_transition_opt_place_indexed_mintrans(padded_real_elems, [eating], sorted([fork_left, thinking, fork_right])))
  
  return make_output(places, transitions, marking, trans_arrays, padded_real_elems)

def gen_conflict_places(n, pad=0):
  places = round_up(n * 2, pad) if pad else n * 2
  transitions = n * 2
  
  # initial tokens in first n places (conflict set)
  initial = list(range(n))
  marking = make_marking(places, initial)
  
  trans_arrays = []
  real_elems = n+1
  padded_real_elems = round_up(real_elems, OPT_TRANS_PAD) 
  conflict_set = list(range(n))
  for i in range(n):
    trans_arrays.extend(make_transition_opt_place_indexed_mintrans(
      padded_real_elems, conflict_set, [n + i]))
  for i in range(n):
    trans_arrays.extend(make_transition_opt_place_indexed_mintrans(
      padded_real_elems, [n + i], conflict_set))
  
  return make_output(places, transitions, marking, trans_arrays, padded_real_elems)

def gen_deadlock(pad=0):
  places = round_up(2, pad) if pad else 2
  transitions = 2

  initial = [0]
  marking = make_marking(places, initial)

  trans_arrays = []
  real_elems = 4
  padded_real_elems = round_up(real_elems, OPT_TRANS_PAD)
  trans_arrays.extend(make_transition_opt_place_indexed_mintrans(padded_real_elems, [0], [1]))
  trans_arrays.extend(make_transition_opt_place_indexed_mintrans(padded_real_elems, [1], [0]))

  return make_output(places, transitions, marking, trans_arrays, padded_real_elems)

# main execution
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("usage: python netgen.py <net_type> <arg> [pad]")
    print("net_type: 0/1=relay, 2=binary_tree, 3=min_cycles, 4=philosophers, 6=conflict")
    sys.exit(1)
  
  net_type = int(sys.argv[1])
  arg = int(sys.argv[2])
  pad = int(sys.argv[3]) if len(sys.argv) > 4 else 0
  
  if arg == 0:
    print("arg must be > 0")
    sys.exit(1)
  
  try:
    if net_type == 0:
      data = gen_relay_race(arg, single_token=True, pad=pad)
    elif net_type == 1:
      data = gen_relay_race(arg, single_token=False, pad=pad)
    elif net_type == 2:
      data = gen_binary_tree(arg, pad=pad)
    elif net_type == 3:
      data = gen_min_cycles(arg, pad=pad)
    elif net_type == 4:
      data = gen_philosophers(arg, pad=pad)
    elif net_type == 6:
      data = gen_conflict_places(arg, pad=pad)
    elif net_type == 7:
      data = gen_deadlock(pad=pad)
    else:
      print(f"unknown net type: {net_type}")
      sys.exit(1)
    
    sys.stdout.buffer.write(data)
    
  except Exception as e:
    print(f"error: {e}", file=sys.stderr)
    sys.exit(1)