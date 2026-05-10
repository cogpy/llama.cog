# OpenCog Cognitive LLM with llama.cpp

This project implements an OpenCog-inspired cognitive architecture integrated with llama.cpp for embodied, architecture-aware language model inference.

## Features

### Core OpenCog Components

- **AtomSpace**: Knowledge representation using nodes and links with truth values and attention values
- **Cognitive Cycle Manager**: Goal-driven processing with perception, goal selection, action selection, execution, and learning phases
- **Embodied Reasoning Engine**: Spatial, temporal, and causal reasoning capabilities
- **Architecture-Aware Inference**: Optimizations for CPU-only, GPU-accelerated, hybrid, and mobile architectures
- **Memory Integration**: Learning from interactions and consolidating knowledge

### Architecture Types

- `CPU_ONLY`: Optimized for CPU-only inference
- `GPU_ACCELERATED`: Utilizes GPU acceleration for improved performance
- `HYBRID_CPU_GPU`: Balances processing between CPU and GPU
- `MOBILE_OPTIMIZED`: Optimized for mobile/edge devices with memory constraints
- `DISTRIBUTED`: Support for distributed inference (placeholder)

## Building

### Prerequisites

- CMake 3.14+
- C++17 compatible compiler
- Optional: CUDA SDK, Metal framework for GPU acceleration

### Build Instructions

```bash
# Configure build
cmake -B build

# Build all components including OpenCog cognitive architecture
cmake --build build --config Release -j $(nproc)
```

### Test the Implementation

```bash
# Run AtomSpace unit tests
g++ -std=c++17 -I include -I src tests/opencog/test_atomspace.cpp src/opencog/atomspace.cpp -o build/test_atomspace
build/test_atomspace
```

## Usage

### Basic Example

```bash
# Run the cognitive LLM system (requires a model file)
build/bin/llama-cognitive -m path/to/model.gguf
```

### Interactive Commands

Once running, the cognitive system supports various commands:

- `help` - Show available commands
- `status` - Display system metrics and status
- `goals` - List active cognitive goals
- `knowledge` - Show current knowledge base
- `add <concept>` - Add a concept to the knowledge base
- `goal <description>` - Set a new cognitive goal
- `spatial <object> <location>` - Add spatial knowledge
- `causal <cause> <effect>` - Add causal relationships
- `plan <goal>` - Generate action plans for goals
- `cycle` - Run a single cognitive cycle
- `continuous` - Start/stop continuous processing
- `save <file>` - Save knowledge to file
- `load <file>` - Load knowledge from file

### Programmatic Usage

```cpp
#include "opencog/opencog.h"

// Create cognitive system
opencog::CognitiveSystem cognitive_system;

// Configure for optimal architecture
opencog::ArchitectureConfig config = opencog::utils::create_optimal_config();

// Initialize with a model
cognitive_system.initialize("model.gguf", config);

// Add knowledge
cognitive_system.add_knowledge("artificial_intelligence", "is_a", "technology");
cognitive_system.add_spatial_knowledge("robot", "laboratory");
cognitive_system.add_causal_knowledge("learning", "improved_performance", 0.9);

// Set goals
cognitive_system.set_goal("Understand user queries", 0.9);

// Process queries
std::string response = cognitive_system.process_query("What is AI?");
```

## Architecture

### Core Classes

#### `AtomSpace`
- Knowledge representation using weighted, labeled graph structure
- Supports nodes (concepts, predicates, variables) and links (relationships)
- Truth values for probabilistic reasoning
- Attention values for importance tracking

#### `CognitiveCycleManager`
- Implements cognitive cycle phases: perception, goal selection, action selection, execution, learning
- Architecture-aware optimizations
- Goal management and prioritization
- Continuous or single-cycle processing

#### `LLMInferenceEngine`
- Integration layer with llama.cpp
- Architecture-aware inference optimization
- Cognitive enhancement of prompts
- Embedding generation and semantic similarity

#### `EmbodiedReasoningEngine`
- Spatial and temporal knowledge representation
- Causal reasoning and consequence inference
- Action planning based on goals
- Feasibility validation for actions

#### `CognitiveSystem`
- Main orchestration class combining all components
- Simple API for knowledge management and query processing
- System monitoring and metrics
- Knowledge persistence

### Knowledge Representation

The system uses an AtomSpace-based knowledge representation:

```
(ConceptNode "cat")
(ConceptNode "animal")
(InheritanceLink
  (ConceptNode "cat")
  (ConceptNode "animal"))
```

Each atom has:
- **Truth Value**: strength (probability) and confidence
- **Attention Value**: importance and urgency for processing

### Cognitive Processing

The system implements a goal-driven cognitive cycle:

1. **Perception**: Process inputs and update attention
2. **Goal Selection**: Choose highest priority goals
3. **Action Selection**: Select appropriate actions
4. **Execution**: Execute actions and generate responses
5. **Learning**: Update knowledge and attention values

## Implementation Details

### AtomSpace Features
- Thread-safe operations with mutex protection
- Efficient indexing by type and name
- Attentional focus for selective processing
- Query capabilities for pattern matching
- Attention decay and memory management

### Architecture Awareness
- Automatic hardware detection and optimization
- Memory constraint management
- Context length optimization based on available resources
- Sampling parameter adjustment for different architectures

### Embodied Reasoning
- Spatial relationships (object locations)
- Temporal sequences (event timing)
- Causal chains (cause-effect relationships)
- Action planning with feasibility checking

## Examples

### Adding Spatial Knowledge
```cpp
// Robot is in laboratory
cognitive_system.add_spatial_knowledge("robot", "laboratory");

// Query spatial relationships
auto consequences = reasoning_engine.infer_consequences("move_robot");
```

### Causal Reasoning
```cpp
// Learning causes improved performance
cognitive_system.add_causal_knowledge("learning", "improved_performance", 0.8);

// Plan actions for a goal
auto actions = cognitive_system.plan_actions("improve robot performance");
// Results: ["identify_knowledge_gap", "search_for_information",
//          "integrate_new_knowledge", "validate_understanding"]
```

### Goal-Driven Processing
```cpp
// Set cognitive goals with priorities
cognitive_system.set_goal("Understand environment", 0.9);
cognitive_system.set_goal("Learn new skills", 0.7);
cognitive_system.set_goal("Help humans", 0.8);

// Start continuous cognitive processing
cognitive_system.start_continuous_processing();
```

## Testing

The system includes comprehensive tests:

- **AtomSpace Tests**: Node/link creation, truth values, attention, queries
- **Cognitive Cycle Tests**: Goal management, processing phases
- **Reasoning Tests**: Spatial, temporal, and causal inference
- **Integration Tests**: End-to-end system functionality

## Performance

The system is designed for real-time cognitive processing:

- **AtomSpace**: O(1) atom retrieval, O(log n) attention sorting
- **Cognitive Cycles**: Configurable frequency (default 10 Hz)
- **Memory Management**: Automatic attention decay and cleanup
- **Architecture Optimization**: Adapts to available hardware resources

## Future Enhancements

- Advanced pattern matching and unification
- More sophisticated learning algorithms
- Distributed cognitive processing
- Integration with robotics frameworks
- Multi-modal perception and reasoning
- Enhanced natural language understanding

## License

This project extends llama.cpp under the MIT License. See LICENSE file for details.
