# FRGS: Long-Horizon UAV Vision-and-Language Navigation with Focused Reasoning Guided by State

Official implementation of the paper:

**FRGS: Long-Horizon UAV Vision-and-Language Navigation with Focused Reasoning Guided by State**

> **Note**
> This repository is being prepared for code release. Some components, scripts, and checkpoints may be cleaned up or reorganized before the final public version.

---

## Overview

Long-horizon UAV vision-and-language navigation (UAV-VLN) remains challenging for zero-shot methods because decision-making over full instructions and globally accumulated history often leads to:

- inaccurate tracking of instruction execution stages,
- weak awareness of subtask completion status,
- interference from redundant historical context,
- unstable navigation decisions in long-horizon scenarios.

To address these issues, we propose **FRGS**, a structured framework for long-horizon zero-shot UAV-VLN. FRGS reformulates navigation from open-ended reasoning over the full context into **stage-wise focused reasoning** guided by the current instruction state.

The key idea is simple:

1. **Instruction State Management (ISM)** decomposes a long instruction into temporally ordered subtasks and tracks which subtask is currently active.
2. **History Context Dynamic Focusing (HCDF)** extracts the historical context that is most relevant to the current subtask from the continuously growing global scene memory.
3. The decision model predicts the next action using only the **current subtask** and its **focused context**, rather than the full instruction and the entire navigation history.

---
## Method Details

### Subtask Graph Construction

A long instruction is transformed into a chained subtask graph:

- nodes represent stage-wise navigation targets,
- edges represent transitions between consecutive subtasks,
- each edge is associated with a constraint set.

The constraints may include:

- direction,
- distance,
- altitude,
- spatial relations with respect to reference landmarks.

### State Tracking

At each time step, FRGS keeps track of the currently active subtask and checks whether all valid constraints are satisfied.  
A minimum-step and maximum-step fault-tolerant mechanism is introduced to reduce premature switching and progress stagnation.

### Focused Historical Context

Historical landmarks are stored in a global scene memory graph.  
FRGS computes relevance scores for nodes based on:

- target consistency,
- reference consistency,
- stage consistency.

It then selects anchor nodes and expands a local subgraph around them to build the focused historical context for the current subtask.

---

## Dataset

We evaluate FRGS on **AirVLN-S**, an aerial vision-and-language navigation benchmark built on Unreal Engine 4.

AirVLN-S contains:

- 25 urban environments,
- more than 870 urban object categories,
- 3,916 UAV flight trajectories with language instructions.

### Data Preparation

Please prepare the dataset according to the official AirVLN-S release.

**TODO**
- Add dataset download link
- Add preprocessing instructions
- Add expected directory structure

Example placeholder structure:

```text
data/
├── AirVLN-S/
│   ├── scenes/
│   ├── annotations/
│   ├── trajectories/
│   └── metadata/