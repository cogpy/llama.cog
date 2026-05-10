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
 * - atom_space: Knowledge representation and storage
 * - cognitive_cycle_manager: goal-driven cognitive processing
 * - llm_inference_engine: Integration with llama.cpp for inference
 * - embodied_reasoning_engine: Spatial and temporal reasoning capabilities
 *
 * Usage Example:
 *
 *   #include "opencog/opencog.h"
 *
 *   // Create cognitive system
 *   auto atomspace = std::make_shared<opencog::atom_space>();
 *   auto llm_engine = std::make_shared<opencog::llm_inference_engine>();
 *   auto cognitive_cycle = std::make_shared<opencog::cognitive_cycle_manager>(atomspace, llm_engine);
 *
 *   // Load model and configure architecture
 *   opencog::architecture_config config;
 *   config.type = opencog::architecture_type::CPU_ONLY;
 *   llm_engine->load_model("model.gguf", config);
 *   cognitive_cycle->set_architecture_config(config);
 *
 *   // Add knowledge and goals
 *   auto concept = atomspace->add_node(opencog::atom_type::CONCEPT_NODE, "AI_system");
 *   cognitive_cycle->add_goal(opencog::goal("Understand user query", 0.8));
 *
 *   // Process input and generate response
 *   cognitive_cycle->process_input("What is artificial intelligence?");
 *   std::string response = cognitive_cycle->generate_response();
 *
 */

// Core OpenCog components
#include "atomspace.h"
#include "cognitive-cycle.h"
#include "llm-inference.h"

#include <thread>

namespace opencog {

/**
 * Main OpenCog Cognitive System class that orchestrates all components
 */
class cognitive_system {
public:
    cognitive_system();
    ~cognitive_system();

    // System initialization
    bool initialize(const std::string& model_path, const architecture_config& config = architecture_config());
    void shutdown();
    bool is_initialized() const;

    // Core cognitive operations
    std::string process_query(const std::string& query);
    void add_knowledge(const std::string& concept_name, const std::string& relation = "", const std::string& target = "");
    void set_goal(const std::string& goal_description, double priority = 0.5);

    // Architecture awareness
    void configure_architecture(const architecture_config& config);
    void optimize_for_hardware();

    // Embodied reasoning
    void add_spatial_knowledge(const std::string& object, const std::string& location);
    void add_causal_knowledge(const std::string& cause, const std::string& effect, double confidence = 0.8);
    std::vector<std::string> plan_actions(const std::string& goal_desc) const;

    // System monitoring
    struct system_metrics {
        size_t atomspace_size;
        size_t active_goals_count;
        size_t cycle_count;
        double last_inference_time_ms;
        double tokens_per_second;
        double memory_usage_mb;
    };

    system_metrics get_metrics() const;

    // Advanced features
    void start_continuous_processing();
    void stop_continuous_processing();
    void save_knowledge(const std::string& filename) const;
    bool load_knowledge(const std::string& filename);

    // Component access (for advanced usage)
    std::shared_ptr<atom_space> get_atomspace() const { return atomspace_; }
    std::shared_ptr<cognitive_cycle_manager> get_cognitive_cycle() const { return cognitive_cycle_; }
    std::shared_ptr<llm_inference_engine> get_llm_engine() const { return llm_engine_; }
    std::shared_ptr<embodied_reasoning_engine> get_reasoning_engine() const { return reasoning_engine_; }

private:
    std::shared_ptr<atom_space> atomspace_;
    std::shared_ptr<llm_inference_engine> llm_engine_;
    std::shared_ptr<cognitive_cycle_manager> cognitive_cycle_;
    std::shared_ptr<embodied_reasoning_engine> reasoning_engine_;
    std::shared_ptr<cognitive_prompt_engine> prompt_engine_;
    std::shared_ptr<cognitive_llm_memory> memory_manager_;

    bool initialized_;
    architecture_config current_config_;
    std::thread cycle_thread_;

    void setup_default_knowledge();
    void initialize_cognitive_components();
};

/**
 * Utility functions for OpenCog system
 */
namespace utils {

    // Architecture detection
    architecture_type detect_optimal_architecture();
    architecture_config create_optimal_config();

    // Knowledge utilities
    std::vector<std::shared_ptr<atom>> parse_knowledge_from_text(const std::string& text, atom_space& atomspace);
    std::string atoms_to_readable_text(const std::vector<std::shared_ptr<atom>>& atoms);

    // Performance utilities
    void benchmark_inference_performance(llm_inference_engine& engine, const std::string& test_prompt);
    void profile_cognitive_cycle(cognitive_cycle_manager& cycle_manager, size_t num_cycles);

} // namespace utils

} // namespace opencog

#endif // OPENCOG_H
