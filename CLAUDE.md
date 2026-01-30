# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training library for large language models (LLMs). It's the open-source implementation of the HybridFlow paper, designed for post-training LLMs with algorithms like PPO, GRPO, RLOO, ReMax, etc.

## Installation & Setup

```bash
# Development installation with test dependencies
pip install -e .[test,vllm]
# or with SGLang
pip install -e .[test,sglang]

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Common Commands

### Linting & Formatting
```bash
# Run all pre-commit checks on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run --all-files ruff
pre-commit run --all-files autogen-trainer-cfg
```

### Running Tests
```bash
# CPU tests (files ending with _on_cpu.py)
pytest tests/**/test_*_on_cpu.py

# GPU unit tests (exclude CPU tests and special directories)
pytest tests/ --ignore=tests/special_distributed --ignore=tests/special_e2e --ignore=tests/special_npu --ignore=tests/special_standalone

# Run a single test file
pytest tests/trainer/test_xxx.py -v

# Run a specific test
pytest tests/trainer/test_xxx.py::test_function_name -v
```

### Training Commands

#### RL Training (PPO/GRPO/RLOO/etc.)
```bash
# Unified RL training entry point
python3 -m verl.trainer.main_ppo <hydra_config_overrides>

# Example: GRPO training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['path/to/train.parquet']" \
    actor_rollout_ref.model.path="path/to/model" \
    trainer.n_gpus_per_node=8

# Example: PPO with vLLM rollout
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.rollout.name=vllm \
    data.train_files="['path/to/train.parquet']" \
    actor_rollout_ref.model.path="path/to/model"

# Example: RLOO algorithm
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files="['path/to/train.parquet']" \
    actor_rollout_ref.model.path="path/to/model"
```

See `examples/<algorithm>_trainer/*.sh` for algorithm-specific examples.

#### Supervised Fine-Tuning
```bash
# Lightweight FSDP-only SFT trainer
python3 -m verl.trainer.fsdp_sft_trainer <hydra_config_overrides>

# Multi-backend SFT trainer with checkpointing
python3 -m verl.trainer.sft_trainer <hydra_config_overrides>

# Ray-based distributed SFT
python3 -m verl.trainer.sft_trainer_ray <hydra_config_overrides>
```

#### Evaluation & Generation
```bash
# Offline evaluation with reward models
python3 -m verl.trainer.main_eval \
    data.val_files="['path/to/test.parquet']" \
    reward_model.enable=true

# Batch generation from datasets
python3 -m verl.trainer.main_generation \
    data.input_files="['path/to/prompts.parquet']" \
    actor_rollout_ref.model.path="path/to/model"

# Start generation server (OpenAI-compatible API)
python3 -m verl.trainer.main_generation_server \
    actor_rollout_ref.model.path="path/to/model" \
    server.port=8000
```

### Building Documentation
```bash
cd docs
pip install -r requirements-docs.txt
make clean && make html
python -m http.server -d _build/html/
```

## Architecture Overview

### Core Components

**Single Controller Architecture** (`verl/single_controller/`)
- Ray-based distributed coordination layer
- `RayWorkerGroup`, `RayResourcePool`: Manage GPU resources across nodes
- `Worker`: Base class for distributed workers with `@register` decorator for RPC methods

**Trainers** (`verl/trainer/`)
- `main_ppo.py`: Entry point for PPO/GRPO training, initializes Ray and runs `RayPPOTrainer`
- `ray_trainer.py`: Orchestrates the RL training loop, manages resource pools and worker groups
- `fsdp_sft_trainer.py`: Standalone SFT trainer using FSDP
- `config/`: Hydra YAML configs with composable defaults (actor, critic, rollout, data, etc.)

**Workers** (`verl/workers/`)
- `fsdp_workers.py`: FSDP-based training workers (actor, critic, reference model)
- `megatron_workers.py`: Megatron-LM backend workers
- `rollout/`: Inference engines (vLLM, SGLang, HF Transformers, TensorRT-LLM)
- `reward_manager/`: Rule-based and model-based reward computation
- `sharding_manager/`: Handle model weight resharding between training and inference

**Protocol** (`verl/protocol.py`)
- `DataProto`: Core data transfer protocol using TensorDict for batched data between workers

**Models** (`verl/models/`)
- Model-specific implementations and weight loaders
- Monkey patches for HuggingFace transformers compatibility

### Data Flow (PPO/GRPO)

1. **Rollout Generation**: Actor model generates responses via vLLM/SGLang
2. **Reward Computation**: Rule-based or model-based rewards
3. **Advantage Estimation**: GAE, GRPO, REINFORCE++, etc. (`trainer/ppo/core_algos.py`)
4. **Policy Update**: Actor/Critic training with FSDP or Megatron-LM
5. **Weight Sync**: 3D-HybridEngine reshards weights between training/inference

### Configuration System

Uses Hydra with composable YAML configs:
- `verl/trainer/config/ppo_trainer.yaml`: Main PPO config with `defaults:` section
- Component configs in subdirectories: `actor/`, `critic/`, `rollout/`, `data/`, etc.
- Override via command line: `python3 -m verl.trainer.main_ppo actor_rollout_ref.actor.optim.lr=1e-6`

### Training Backends

- **FSDP/FSDP2**: PyTorch native distributed training
- **Megatron-LM**: For large models with tensor/pipeline parallelism

### Inference Backends

- **vLLM** (`workers/rollout/vllm_rollout/`)
  - High-performance async server-based generation
  - Token/sequence parallelism support
  - LoRA inference
  - FP8 quantization
  - Best for: High-throughput batch inference

- **SGLang** (`workers/rollout/sglang_rollout/`)
  - Radix attention engine with prefix caching
  - Async HTTP server adapter
  - Multi-turn conversation support
  - Best for: Agent interactions, multi-turn dialogue

- **TensorRT-LLM** (`workers/rollout/trtllm_rollout/`)
  - NVIDIA's optimized inference engine
  - Multi-node deployment support
  - GPU memory optimization
  - Best for: Production deployment, maximum throughput

- **HF Transformers** (`workers/rollout/hf_rollout.py`)
  - Direct Hugging Face integration
  - Simple setup, no server needed
  - Best for: Quick prototyping, small-scale experiments

- **Naive Rollout** (`workers/rollout/naive/`)
  - Single-GPU reference implementation
  - Best for: Debugging, understanding rollout logic

## Test Organization

- `tests/trainer/`, `tests/models/`, etc.: Unit tests mirroring `verl/` structure
- `tests/special_distributed/`: Multi-GPU tests
- `tests/special_e2e/`: End-to-end training tests
- `tests/special_npu/`: Ascend NPU tests
- `tests/special_sanity/`: Quick sanity checks
- Files ending with `_on_cpu.py` run on CPU only

## Key Algorithms

Supported RL algorithms (configured via `algorithm.adv_estimator`). All algorithms use the same entry point (`main_ppo`) with different estimator configurations:

### Core Algorithms

| Algorithm | Estimator | Description | Use Case |
|-----------|-----------|-------------|----------|
| **PPO** | `gae` | Generalized Advantage Estimation | Classic RL, requires critic |
| **GRPO** | `grpo` | Group Relative Policy Optimization | Outcome-based, no critic needed |
| **RLOO** | `rloo` | REINFORCE Leave-One-Out | Baseline-free variance reduction |
| **ReMax** | `remax` | Reward Maximization | Direct reward optimization |
| **OPO** | `opo` | Outcome-only Policy Optimization | Simplified outcome-based |
| **GPG** | `gpg` | Group Policy Gradient | Group-based policy updates |

### Algorithm Variants

- `grpo_vectorized`: Optimized GRPO with vectorized operations
- `grpo_passk`: GRPO with Pass@k evaluation (second-best reward comparison)
- `reinforce_plus_plus`: Policy gradient with variance reduction
- `reinforce_plus_plus_baseline`: REINFORCE++ with baseline subtraction
- `rloo_vectorized`: Optimized RLOO implementation
- `optimal_token_baseline`: Advanced token-level baseline
- `tir_optimal_token_baseline`: Time-independent reward baseline

### Advanced Features

- **KL Penalty Control**: Fixed or adaptive KL penalties (via `algorithm.kl_ctrl`)
- **Rollout Correction**: Off-policy correction with 15+ presets (importance sampling, rejection sampling)
- **Preference Feedback PPO** (`pf_ppo`): DPO-style preference learning
- **Advantage Normalization**: Sequence-level or token-level normalization

### Example Usage

```bash
# PPO with GAE
python3 -m verl.trainer.main_ppo algorithm.adv_estimator=gae

# GRPO (no critic needed)
python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo

# RLOO with rollout correction
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    algorithm.rollout_correction.enable=true
```

See `verl/trainer/ppo/core_algos.py` for implementation details.

## Examples Directory Structure

The `examples/` directory contains ready-to-run training scripts organized by algorithm and use case:

### By RL Algorithm
- `ppo_trainer/` - Classic PPO with GAE (various backends)
- `grpo_trainer/` - GRPO implementations (50+ scripts for different models)
- `rloo_trainer/` - REINFORCE Leave-One-Out
- `remax_trainer/` - ReMax algorithm
- `reinforce_plus_plus_trainer/` - REINFORCE++ with/without baselines
- `gspo_trainer/`, `gmpo_trainer/`, `gpg_trainer/`, `sapo_trainer/`, `otb_trainer/`, `cispo_trainer/` - Specialized RL variants
- `mtp_trainer/` - MTP (Megatron Training-inference Parallelism)

### By Training Type
- `sft/` - Supervised fine-tuning examples
- `generation/` - Standalone generation scripts
- `sglang_multiturn/` - Multi-turn conversation with SGLang

### Advanced Features
- `rollout_correction/` - Off-policy correction examples
- `router_replay/` - Router replay training
- `prefix_grouper/` - Prefix grouping optimization
- `split_placement/` - Advanced GPU placement strategies
- `tuning/` - Hyperparameter tuning scripts
- `tutorial/` - Getting started guides

### Data Preprocessing
- `data_preprocess/` - Data preparation scripts
  - `gsm8k.py`, `math_dataset.py` - Math datasets
  - `gsm8k_multiturn_w_tool.py` - Tool-calling datasets
  - Various domain-specific preprocessors

### Deployment
- `slurm/` - SLURM cluster submission scripts
- `skypilot/` - Sky computing platform integration
- `ray/` - Ray cluster examples

## Recent Features & Improvements

### TensorRT-LLM Integration (2024)
- Added TensorRT-LLM as rollout engine for high-performance inference
- Multi-node deployment support
- Async server architecture

### Rollout Correction System
- Off-policy correction with 15+ preset configurations
- Importance sampling and rejection sampling
- Handles distribution shift during training

### Advanced Metrics
- Preemption metrics for rollout stability
- Timing and throughput tracking
- Variance analysis for advantage estimation

### Multi-Turn & Agent Support
- SGLang async policy support for agent interactions
- Tool-calling framework with customizable tools
- ReAct-style reasoning loops

### Performance Optimizations
- Nested tensor dispatch for 3+ dimensional tensors
- Rollout and validation parallel processing
- Sequence length balancing
- Model weight resharding between training/inference

## Recipe Submodule

The `recipe/` directory is a git submodule pointing to [verl-recipe](https://github.com/verl-project/verl-recipe), containing advanced training recipes and research implementations.

### Initialize Recipe
```bash
git submodule update --init --recursive recipe
```

### Available Recipes

The recipe repository contains 20+ advanced training recipes organized by research area:

#### Core Algorithm Recipes
- **`prime/`** - PRIME algorithm implementation
- **`r1/`** - R1 reasoning algorithm (CPU/GPU)
- **`r1_ascend/`** - R1 for Ascend NPU
- **`entropy/`** - Entropy-based exploration
- **`flowrl/`** - Flow-based RL

#### Preference Learning
- **`dapo/`** - Distribution-Aware Preference Optimization
- **`fapo/`** - Fast Approximate Policy Optimization
- **`spo/`** - Simplified Preference Optimization
- **`sppo/`** - Stepwise Preference Policy Optimization

#### Advanced Techniques
- **`specRL/`** - Speculative RL training
- **`spin/`** - Self-Play with Intrinsic Motivation
- **`gkd/`** - Generalized Knowledge Distillation
- **`retool/`** - Reasoning with Tools
- **`open_math_reasoning/`** - Mathematical reasoning tasks

#### Multi-Agent & Collaborative
- **`collabllm/`** - Collaborative LLM training
- **`langgraph_agent/`** - LangGraph agent integration

#### Domain-Specific
- **`minicpmo/`** - MiniCPM-Omni multimodal training
- **`deepeyes/`** - Vision-language tasks
- **`infigui-g1/`** - GUI interaction agent
- **`char_count/`** - Character counting task (tutorial)

#### Remote Execution
- **`genrm_remote/`** - Remote generation with reward models

Each recipe includes:
- Training scripts with optimized hyperparameters
- Data preprocessing pipelines
- Model-specific configurations
- Evaluation scripts
- README with usage instructions

### Example: Running a Recipe

```bash
cd recipe/r1
bash run_qwen2.5_r1.sh  # Run R1 training on Qwen2.5

cd recipe/dapo
bash train_llama3_dapo.sh  # DAPO training on Llama3
```

## Learning Resources

### Comprehensive Learning Notes (`learning_notes/`)

This repository includes **210,000+ words** of detailed learning materials covering the complete verl training pipeline. All materials are in Chinese (中文) and provide in-depth technical analysis with source code references.

#### **Part 1: Quick Start** (`01_快速上手/`)
- **Content**: 28,000+ words (guides + deep-dives)
- **Files**:
  - `01_快速上手.md` - Getting started guide (8,000+ words)
  - `ray_trainer_详解.md` - RayPPOTrainer deep-dive (12,000+ words)
  - `配置系统详解.md` - Hydra configuration system (8,000+ words)
  - `check_env.sh`, `check_data.py`, `run_first_training.sh` - Utility scripts
- **Topics**:
  - Environment setup and validation
  - 7-stage training pipeline
  - RayPPOTrainer architecture (WorkerGroups, ResourcePools)
  - Hydra configuration system
  - First training run and monitoring

#### **Part 2: Data Preparation** (`02_数据准备/`)
- **Content**: 25,000+ words
- **Files**:
  - `02_数据准备.md` - Data format guide (10,000+ words)
  - `reward_系统详解.md` - Reward system deep-dive (15,000+ words)
  - `data_quality_check.py` - Data validation script
- **Topics**:
  - Parquet format and 4 prompt templates
  - RewardManager architecture and call flow
  - GSM8K reward implementation
  - Single-turn and multi-turn data
  - Custom reward functions
  - **NEW**: Reward token placement mechanism (outcome supervision)
  - **NEW**: GAE vs GRPO advantage broadcasting (recursive vs explicit)

#### **Part 3: RL Algorithms** (`03_RL算法/`)
- **Content**: 50,000+ words
- **Files**:
  - `03_RL算法概览.md` - Algorithm overview and decision tree
  - `GRPO_详解.md` - GRPO deep-dive with source code (15,000+ words)
  - `PPO_详解.md` - PPO deep-dive with GAE + Clipping (20,000+ words)
- **Topics**:
  - GRPO, PPO, RLOO, ReMax algorithm principles
  - GRPO group mechanism and advantage calculation
  - PPO's GAE and clipping mechanism
  - Algorithm selection guide
  - Configuration switching
  - **NEW**: old_log_prob vs new_log_prob (importance sampling)
  - **NEW**: ref_log_probs calculation and KL dual penalty (reward vs loss)

#### **Part 4: Reward Design** (`04_Reward设计/`)
- **Content**: 40,000+ words
- **Files**:
  - `自定义Reward实践指南.md` - Custom reward practical guide
- **Topics**:
  - Reward function fundamentals and design principles
  - RewardManager complete call flow (source-level)
  - 3 reward types: Rule-based, Model-based, Sandbox
  - 10+ practical examples (simple to complex)
  - Reward shaping and multi-objective optimization
  - Complete debugging and validation workflow

#### **Part 5: Agent RL** (`05_Agent_RL/`)
- **Content**: 45,000+ words
- **Files**:
  - `Agent_Loop详解.md` - Agent Loop system deep-dive
  - `README.md` - Agent RL overview and learning path
- **Topics**:
  - Agent Loop architecture (AsyncLLMServerManager + Workers)
  - Tool calling mechanism and lifecycle (create → execute → calc_reward → release)
  - Token-based API vs Chat Completion API
  - Multi-turn trajectory consistency guarantee
  - response_mask mechanism (loss calculation, advantage broadcasting)
  - End-to-end training flow (GSM8K Tool Agent)
  - Debugging techniques (reward=0, PPO ratio explosion, response_mask errors)
  - Best practices (system prompt design, tool design principles, reward shaping)
  - Hyperparameter tuning guide

### Learning Path Overview

```
Day 1-2:  Part 1 - Quick Start (Environment + First Training)
Day 3-4:  Part 2 - Data Preparation (Formats + Rewards)
Day 5-7:  Part 3 - RL Algorithms (PPO/GRPO)
Day 8:    Part 4 - Reward Design (Custom Functions)
Day 9-13: Part 5 - Agent RL (Tool Calling + Multi-turn)
```

### Key Technical Deep-Dives

The learning notes include detailed Q&A sections addressing common technical questions:

1. **RayPPOTrainer Training Loop** (Part 1)
   - 7-stage pipeline with code references
   - WorkerGroup and ResourcePool management
   - Weight synchronization between training/inference

2. **Reward System Architecture** (Part 2)
   - 4 types of RewardManager (Naive, Batch, Loop, DAPO)
   - Reward token placement (why last token only?)
   - GAE vs GRPO advantage broadcasting mechanisms

3. **PPO Technical Details** (Part 3)
   - old_log_prob vs new_log_prob (proximal anchor vs optimization target)
   - Importance sampling ratio calculation
   - ref_log_probs and KL dual penalty (use_kl_in_reward vs use_kl_loss)
   - Three policy modes: Decoupled, Bypass, Rollout Correction

4. **Custom Reward Implementation** (Part 4)
   - Reward function signature and RewardManager integration
   - 10+ practical examples with complete code
   - Debugging workflow and validation

5. **Agent Loop System** (Part 5)
   - Tool lifecycle management
   - Multi-turn trajectory consistency
   - response_mask computation and usage
   - Common pitfalls and solutions

### Accessing Learning Materials

All learning materials are located in the `learning_notes/` directory:

```bash
# View table of contents
cat learning_notes/README.md

# Start with Part 1
cd learning_notes/01_快速上手
cat README.md

# Check environment
bash check_env.sh

# Read deep-dive materials
cat ray_trainer_详解.md
cat 配置系统详解.md
```

### Relationship to CLAUDE.md

- **CLAUDE.md** (this file): Developer-facing technical overview, architecture, and command reference (English)
- **learning_notes/**: Comprehensive learning materials with step-by-step guides, source code analysis, and practical examples (Chinese)
- **LEARNING_GUIDE.md**: Complete learning roadmap with 7 sections covering both application and deep technical analysis (Chinese)

Together, these resources provide complete coverage from quick start to advanced customization.
