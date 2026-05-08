#ifndef OPENCOG_H
#define OPENCOG_H

/**
 * OpenCog Cognitive LLM Inference Engine
 *
 * This header provides the main interface for the OpenCog-inspired cognitive
 * architecture integrated with llama.cpp for embodied, architecture-aware
 * language model inference.
 *
 * Key Components:
 * - AtomSpace: Knowledge representation and storage
 * - CognitiveCycleManager: Goal-driven cognitive processing
 * - LLMInferenceEngine: Integration with llama.cpp for inference
 * - EmbodiedReasoningEngine: Spatial and temporal reasoning capabilities
 *
 * Usage Example:
 *
 *   #include "opencog/opencog.h"
 *
 *   // Create cognitive system
 *   auto atomspace = std::make_shared<opencog::AtomSpace>();
 *   auto llm_engine = std::make_shared<opencog::LLMInferenceEngine>();
 *   auto cognitive_cycle = std::make_shared<opencog::CognitiveCycleManager>(atomspace, llm_engine);
 *
 *   // Load model and configure architecture
 *   opencog::ArchitectureConfig config;
 *   config.type = opencog::ArchitectureType::CPU_ONLY;
 *   llm_engine->load_model("model.gguf", config);
 *   cognitive_cycle->set_architecture_config(config);
 *
 *   // Add knowledge and goals
 *   auto concept = atomspace->add_node(opencog::AtomType::CONCEPT_NODE, "AI_system");
 *   cognitive_cycle->add_goal(opencog::Goal("Understand user query", 0.8));
 *
 *   // Process input and generate response
 *   cognitive_cycle->process_input("What is artificial intelligence?");
 *   std::string response = cognitive_cycle->generate_response();
 *
 */

// Core OpenCog components
#include "atomspace.h"
#include "cognitive_cycle.h"
#include "llm_inference.h"

#include <thread>

namespace opencog {

/**
 * Main OpenCog Cognitive System class that orchestrates all components
 */
class CognitiveSystem {
public:
    CognitiveSystem();
    ~CognitiveSystem();

    // System initialization
    bool initialize(const std::string& model_path, const ArchitectureConfig& config = ArchitectureConfig());
    void shutdown();
    bool is_initialized() const;

    // Core cognitive operations
    std::string process_query(const std::string& query);
    void add_knowledge(const std::string& concept_name, const std::string& relation = "", const std::string& target = "");
    void set_goal(const std::string& goal_description, double priority = 0.5);

    // Architecture awareness
    void configure_architecture(const ArchitectureConfig& config);
    void optimize_for_hardware();

    // Embodied reasoning
    void add_spatial_knowledge(const std::string& object, const std::string& location);
    void add_causal_knowledge(const std::string& cause, const std::string& effect, double confidence = 0.8);
    std::vector<std::string> plan_actions(const std::string& goal) const;

    // System monitoring
    struct SystemMetrics {
        size_t atomspace_size;
        size_t active_goals_count;
        size_t cycle_count;
        double last_inference_time_ms;
        double tokens_per_second;
        double memory_usage_mb;
    };

    SystemMetrics get_metrics() const;

    // Advanced features
    void start_continuous_processing();
    void stop_continuous_processing();
    void save_knowledge(const std::string& filename) const;
    bool load_knowledge(const std::string& filename);

    // Component access (for advanced usage)
    std::shared_ptr<AtomSpace> get_atomspace() const { return atomspace_; }
    std::shared_ptr<CognitiveCycleManager> get_cognitive_cycle() const { return cognitive_cycle_; }
    std::shared_ptr<LLMInferenceEngine> get_llm_engine() const { return llm_engine_; }
    std::shared_ptr<EmbodiedReasoningEngine> get_reasoning_engine() const { return reasoning_engine_; }

private:
    std::shared_ptr<AtomSpace> atomspace_;
    std::shared_ptr<LLMInferenceEngine> llm_engine_;
    std::shared_ptr<CognitiveCycleManager> cognitive_cycle_;
    std::shared_ptr<EmbodiedReasoningEngine> reasoning_engine_;
    std::shared_ptr<CognitivePromptEngine> prompt_engine_;
    std::shared_ptr<CognitiveLLMMemory> memory_manager_;

    bool initialized_;
    ArchitectureConfig current_config_;
    std::thread cycle_thread_;

    void setup_default_knowledge();
    void initialize_cognitive_components();
};

/**
 * Utility functions for OpenCog system
 */
namespace utils {

    // Architecture detection
    ArchitectureType detect_optimal_architecture();
    ArchitectureConfig create_optimal_config();

    // Knowledge utilities
    std::vector<std::shared_ptr<Atom>> parse_knowledge_from_text(const std::string& text, AtomSpace& atomspace);
    std::string atoms_to_readable_text(const std::vector<std::shared_ptr<Atom>>& atoms);

    // Performance utilities
    void benchmark_inference_performance(LLMInferenceEngine& engine, const std::string& test_prompt);
    void profile_cognitive_cycle(CognitiveCycleManager& cycle_manager, size_t num_cycles);

} // namespace utils

} // namespace opencog

#endif // OPENCOG_H
