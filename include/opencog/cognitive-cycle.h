#ifndef OPENCOG_COGNITIVE_CYCLE_H
#define OPENCOG_COGNITIVE_CYCLE_H

#include "atomspace.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

namespace opencog {

// Forward declarations
class llm_inference_engine;
class goal_manager;
class perception_processor;

// Goal representation for cognitive processing
struct goal_t {
    std::string description;
    double priority;
    std::chrono::steady_clock::time_point created;
    std::shared_ptr<Atom> goal_atom;
    bool completed;

    goal_t(const std::string& desc, double prio = 0.5)
        : description(desc), priority(prio), created(std::chrono::steady_clock::now()), completed(false) {}

    bool operator<(const goal_t& other) const {
        return priority < other.priority; // Max heap based on priority
    }
};

// Cognitive state tracking
struct cognitive_state {
    std::shared_ptr<atom_space> atomspace;
    std::priority_queue<goal_t> active_goals;
    std::vector<std::shared_ptr<Atom>> current_focus;
    std::string current_context;
    size_t cycle_count;
    std::chrono::steady_clock::time_point last_cycle;

    cognitive_state(std::shared_ptr<atom_space> as)
        : atomspace(as), cycle_count(0), last_cycle(std::chrono::steady_clock::now()) {}
};

// Architecture awareness for different hardware/model configurations
enum class architecture_type {
    CPU_ONLY,
    GPU_ACCELERATED,
    HYBRID_CPU_GPU,
    DISTRIBUTED,
    MOBILE_OPTIMIZED
};

struct architecture_config {
    architecture_type type;
    size_t memory_limit_mb;
    size_t max_context_length;
    double inference_budget_ms;  // Time budget per inference
    bool enable_speculative_decoding;

    architecture_config()
        : type(architecture_type::CPU_ONLY), memory_limit_mb(4096),
          max_context_length(2048), inference_budget_ms(100.0),
          enable_speculative_decoding(false) {}
};

// Core cognitive cycle manager
class cognitive_cycle_manager {
public:
    cognitive_cycle_manager(std::shared_ptr<atom_space> atomspace,
                         std::shared_ptr<llm_inference_engine> llm_engine);
    ~cognitive_cycle_manager() = default;

    // Cognitive cycle execution
    void run_single_cycle();
    void run_continuous(size_t max_cycles = 0);
    void stop();

    // Goal management
    void add_goal(const goal_t& goal);
    void remove_goal(const std::string& description);
    std::vector<goal_t> get_active_goals() const;

    // Architecture awareness
    void set_architecture_config(const architecture_config& config);
    const architecture_config& get_architecture_config() const;

    // Perception and action
    void process_input(const std::string& input);
    std::string generate_response(const std::string& context = "");

    // Returns a snapshot of the cognitive state. Returned by value (rather
    // than by reference) so callers cannot observe partial mutations from
    // the continuous-processing worker thread; the snapshot is taken under
    // state_mutex_.
    cognitive_state get_state() const;

    // Configuration
    void set_cycle_frequency_hz(double frequency);
    void set_attention_decay_rate(double rate);

private:
    // state_ is accessed concurrently from the main thread (process_input,
    // generate_response, add_goal, get_active_goals, get_state) and from the
    // worker thread launched by run_continuous(). All accesses must be
    // serialised through state_mutex_.
    mutable std::mutex state_mutex_;
    cognitive_state state_;
    std::shared_ptr<llm_inference_engine> llm_engine_;
    architecture_config arch_config_;

    std::atomic<bool> running_;
    double cycle_frequency_hz_;
    double attention_decay_rate_;

    // Cognitive cycle phases
    void perception_phase();
    void goal_selection_phase();
    void action_selection_phase();
    void execution_phase();
    void learning_phase();

    // Architecture-aware optimizations
    void optimize_for_architecture();
    void manage_memory_constraints();
    void balance_inference_load();

    // Internal utilities
    void update_attentional_focus();
    void process_goal_completion();
    std::vector<std::shared_ptr<Atom>> select_relevant_knowledge(const std::string& context) const;
};

// Embodied reasoning capabilities
class embodied_reasoning_engine {
public:
    embodied_reasoning_engine(std::shared_ptr<atom_space> atomspace);
    ~embodied_reasoning_engine() = default;

    // Spatial and temporal reasoning
    void add_spatial_knowledge(const std::string& object, const std::string& location);
    void add_temporal_knowledge(const std::string& event, const std::string& time_context);

    // Causal reasoning
    void add_causal_relation(const std::string& cause, const std::string& effect, double confidence = 0.8);
    std::vector<std::string> infer_consequences(const std::string& action) const;

    // Contextual understanding
    void update_context(const std::string& new_context);
    std::string get_current_context() const;

    // Embodied action planning
    std::vector<std::string> plan_actions(const std::string& goal) const;
    bool validate_action_feasibility(const std::string& action, const std::string& context) const;

private:
    std::shared_ptr<atom_space> atomspace_;
    std::string current_context_;

    // Internal reasoning methods
    std::shared_ptr<Atom> create_spatial_atom(const std::string& object, const std::string& location);
    std::shared_ptr<Atom> create_temporal_atom(const std::string& event, const std::string& time);
    std::shared_ptr<Atom> create_causal_atom(const std::string& cause, const std::string& effect);
};

} // namespace opencog

#endif // OPENCOG_COGNITIVE_CYCLE_H
