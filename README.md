# gpu-petri-nets

**Master's Thesis — University of Hamburg, mid-2025**
This repository shows my master's thesis submitted in mid-2025 in fulfillment of the degree of Master of Science (M.Sc.) at the University of Hamburg.

GPU-based simulation and state space analysis tools for Petri nets, developed as part of a master's thesis on CUDA-accelerated Petri net algorithms in monolithic and distributed systems.

The whole development process may be seen in our department's git.

A German-language explanation video is available here: https://www.youtube.com/watch?v=NjHkXLm_FDQ

## Abstract

This thesis addresses the development of GPU-based applications for the simulation and state space analysis of Petri nets. Petri nets are a well-established model for describing concurrent systems, yet their analysis quickly reaches computational limits as size and complexity grow. Traditional CPU-based simulators and analysis tools are insufficiently scalable due to hardware constraints.

By leveraging NVIDIA's CUDA framework, the inherent parallelism of Petri nets is mapped onto modern GPU architectures. This enables the use of execution and memory hierarchies, as well as domain-specific optimization strategies to improve runtime performance. The work follows a data-oriented programming approach, developing prototypes iteratively, each of which is analyzed, optimized, and extended. Various distributed approaches are developed, accompanied by a thorough investigation of distribution strategies and the communication frameworks employed, such as ZeroMQ, MPI, and NVSHMEM.

A particular focus is placed on comparing monolithic and distributed architectures. While monolithic systems rely on a single GPU, distributed systems allow for load distribution across multiple computing nodes, overcoming memory and computational limitations of individual devices. The study investigates how GPU-based parallelization can be further enhanced through such distribution.

To demonstrate practical applicability, the developed simulator prototypes are also integrated into an existing, established Petri net editor.

The results show that GPU-based simulations achieve significant performance gains compared to CPU-based approaches. They also provide fundamental insights into parallelization strategies, memory optimizations, and the potential transfer of GPU techniques to CPU-based methods, contributing both to the theoretical advancement of Petri net simulation and to its practical applicability in large-scale scenarios. Additionally, the thesis shows how GPU-based state space analysis can be optimized to achieve similarly significant performance improvements.

## Overview

This repository contains high-performance GPU-based simulation and analysis tools for various Petri net formalisms (Place-Transition, Boolean, Arc-Constant Colored).

### Standalone executables

Three standalone source files can be compiled into the following tools:

- `simulator.cu`: Local and distributed simulation of Place-Transition, Boolean, and Arc-Constant (colored) nets. Supports place-distributed and transition-distributed simulation in addition to local execution.
- `local_rg_gen.cu`: Local Reachability Graph generation.
- `dist_rg_gen.cu`: Distributed Reachability Graph generation using NVSHMEM/MPI.

A Petri net editor integration is also included.

Many additional prototypes, net generation scripts, and execution helper scripts are also included.

## Requirements

- CUDA Toolkit (tested with Compute Capability <= 8.0, CUDA 11/12/13) and an NVIDIA GPU
- ZeroMQ (`libzmq`) — required for `local_rg_gen` and the standard simulator
- NVSHMEM & HPC-X / MPI — required for `dist_rg_gen`
- Java JDK — required for JNI-based builds
- Python 3 — for net generation and evaluation scripts

## Compilation

The Makefile handles the various build configurations.

After base compilation, refer to the manual given by running:

```bash
executable -h
```

This will print a detailed description of usage and options.

### Standard build

To build the standard simulator:

```bash
make simulator
```

To build the Reachability Graph generators:

```bash
make local_rg_gen
make dist_rg_gen
```

### Compilation flags

- `-DDEBUG` enables debug mode, which includes debug prints.
- For more flags, see the top of each `.cu` file.

### Debug build

Append `-debug` to the target name to enable debug prints and symbols:

```bash
make simulator-debug
make local_rg_gen-debug
```

### Distributed build (NVSHMEM)

For distributed tools, use the NVSHMEM targets:

```bash
make dist_rg_gen-nvshmem
```

## Usage

### 1. Simulator

```bash
# Interactive mode (steps=0)
./simulator 0 <file-in> <file/pipe-out>

# Fixed steps
./simulator 1000 <file-in> <file-out>

# Distributed / Remote Mode
# Receiver:
./simulator 0 input.ckk out --remote 999 localhost:998
# Sender/Partner:
./simulator 0 input.ckk out --remote 998 localhost:999
```

### 2. Local Reachability Graph CLI

```bash
./local_rg_gen <file-in> <nodes> <arcs> <file-out>
```

- `nodes`: Number of nodes/states to allocate.
- `arcs`: Number of arcs to allocate.

### 3. Distributed Reachability Graph

Use the helper script to launch with the MPI/NVSHMEM configuration properly set (GPU affinity, etc.):

```bash
./eval_helpers/dist_rg_launch.sh <file-in> <nodes> <arcs> <file-out>

# For multi-node:
./eval_helpers/dist_rg_launch.sh --hostfile <hostfile> <file-in> <nodes> <arcs> <file-out>
```

### 4. Net generation (Python)

Scripts in `net_generators/` create benchmark nets compatible with the simulator. Each script requires different parameters; run it without arguments to see its usage text.

Example for PT-nets:

```bash
python3 net_generators/pt_gen.py <net_type_id> <size_arg> > my_net.cpt
```

Net types (not every net is supported by every generator):

- `0`: Relay Race (single token)
- `1`: Relay Race
- `2`: Binary Tree
- `3`: Min Cycles
- `4`: Philosophers
- `6`: Conflict
- `7`: Generator Net

#### File formats

There are no distinct file formats, since file types are distinguished by a magic number stored in the metadata section of the net.