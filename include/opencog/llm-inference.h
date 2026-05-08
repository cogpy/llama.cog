#ifndef OPENCOG_LLM_INFERENCE_H
#define OPENCOG_LLM_INFERENCE_H

#include "llama.h"
#include "atomspace.h"
#include "cognitive-cycle.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace opencog {

// Integration layer between OpenCog and llama.cpp
class llm_inference_engine {
public:
    llm_inference_engine();
    ~llm_inference_engine();

    // Model management
    bool load_model(const std::string& model_path, const architecture_config& config);
    void unload_model();
    bool is_model_loaded() const;

    // Core inference with cognitive enhancement
    std::string generate_response(const std::string& prompt,
                                const std::vector<std::shared_ptr<Atom>>& context_atoms = {},
                                size_t max_tokens = 256);

    // Architecture-aware inference
    std::string cognitive_inference(const std::string& input,
                                  const cognitive_state& state,
                                  const architecture_config& config);

    // Reasoning-enhanced generation
    std::string reason_and_generate(const std::string& query,
                                  const std::vector<std::shared_ptr<Atom>>& knowledge_base);

    // Embeddings for semantic similarity
    std::vector<float> get_embeddings(const std::string& text);
    double compute_semantic_similarity(const std::string& text1, const std::string& text2);

    // Context management
    void update_context(const std::vector<std::shared_ptr<Atom>>& atoms);
    void clear_context();

    // Performance monitoring
    struct inference_metrics {
        double tokens_per_second;
        double memory_usage_mb;
        double processing_time_ms;
        size_t tokens_generated;
        size_t context_length;
    };

    inference_metrics get_last_metrics() const;

    // Configuration
    void set_sampling_params(float temperature = 0.8f, float top_p = 0.9f, int top_k = 40);
    void set_architecture_optimization(const architecture_config& config);

private:
    llama_model   * model_;
    llama_context * context_;
    llama_sampler * sampler_;

    bool model_loaded_;
    // Tracks whether llama_backend_init() has been paired with a matching
    // llama_backend_free(). Without this flag, a failed load_model() (which
    // already initialised the backend) followed by destruction would call
    // llama_backend_free() twice -- once in the failure path, once via
    // unload_model() in the destructor.
    bool backend_initialized_ = false;
    inference_metrics last_metrics_;
    architecture_config current_config_;

    // Cognitive enhancement methods
    std::string atoms_to_context_string(const std::vector<std::shared_ptr<Atom>>& atoms) const;
    std::string format_prompt_with_reasoning(const std::string& prompt,
                                           const std::vector<std::shared_ptr<Atom>>& reasoning_atoms) const;

    // Architecture optimization
    void optimize_for_cpu();
    void optimize_for_gpu();
    void optimize_for_hybrid();
    void optimize_for_mobile();

    // Performance monitoring
    void update_metrics(const std::chrono::steady_clock::time_point& start_time,
                       size_t tokens_generated);
};

// Cognitive prompt engineering
class cognitive_prompt_engine {
public:
    cognitive_prompt_engine(std::shared_ptr<atom_space> atomspace);
    ~cognitive_prompt_engine() = default;

    // Prompt enhancement with cognitive context
    std::string enhance_prompt(const std::string& base_prompt,
                             const std::vector<std::shared_ptr<Atom>>& relevant_atoms,
                             const std::string& reasoning_context = "") const;

    // Goal-directed prompting
    std::string create_goal_prompt(const goal_t& goal,
                                 const std::vector<std::shared_ptr<Atom>>& knowledge) const;

    // Chain-of-thought reasoning prompts
    std::string create_reasoning_chain(const std::string& problem,
                                     const std::vector<std::shared_ptr<Atom>>& facts) const;

    // Embodied reasoning prompts
    std::string create_embodied_prompt(const std::string& action,
                                     const std::string& environment_context,
                                     const std::vector<std::shared_ptr<Atom>>& spatial_knowledge) const;

private:
    std::shared_ptr<atom_space> atomspace_;

    std::string format_atoms_as_knowledge(const std::vector<std::shared_ptr<Atom>>& atoms) const;
    std::string create_reasoning_template() const;
    std::string create_embodied_template() const;
};

// Memory integration with atom_space
class cognitive_llm_memory {
public:
    cognitive_llm_memory(std::shared_ptr<atom_space> atomspace);
    ~cognitive_llm_memory() = default;

    // Memory encoding and retrieval
    void encode_interaction(const std::string& input, const std::string& output);
    void encode_reasoning_step(const std::string& premise, const std::string& conclusion);

    std::vector<std::shared_ptr<Atom>> retrieve_relevant_memories(const std::string& query,
                                                                size_t max_memories = 10) const;

    // Learning from interactions
    void learn_from_feedback(const std::string& interaction, bool positive_feedback);
    void update_concept_strengths(const std::vector<std::string>& concepts, double delta);

    // Memory management
    void consolidate_memories();
    void decay_old_memories(double decay_rate = 0.01);

private:
    std::shared_ptr<atom_space> atomspace_;

    std::shared_ptr<Atom> create_interaction_atom(const std::string& input, const std::string& output);
    std::shared_ptr<Atom> create_reasoning_atom(const std::string& premise, const std::string& conclusion);
    double compute_relevance_score(std::shared_ptr<Atom> memory, const std::string& query) const;
};

} // namespace opencog

#endif // OPENCOG_LLM_INFERENCE_H
