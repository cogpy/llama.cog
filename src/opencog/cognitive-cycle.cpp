#include "opencog/cognitive-cycle.h"
#include "opencog/llm-inference.h"
#include <thread>
#include <algorithm>
#include <sstream>

namespace opencog {

// cognitive_cycle_manager implementation
cognitive_cycle_manager::cognitive_cycle_manager(std::shared_ptr<atom_space> atomspace,
                                           std::shared_ptr<llm_inference_engine> llm_engine)
    : state_(atomspace), llm_engine_(llm_engine), running_(false),
      cycle_frequency_hz_(10.0), attention_decay_rate_(0.01) {
}

void cognitive_cycle_manager::run_single_cycle() {
    auto cycle_start = std::chrono::steady_clock::now();

    // The phase methods and the cycle_count/last_cycle updates all touch
    // state_, which is also read/written by the main thread via
    // process_input, generate_response, add_goal, etc. Hold state_mutex_ for
    // the duration of the work, but release it before sleep_for so other
    // threads aren't blocked while we wait to maintain the target frequency.
    size_t cycle_count_snapshot = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Standard cognitive cycle phases
        perception_phase();
        goal_selection_phase();
        action_selection_phase();
        execution_phase();
        learning_phase();

        // Update state
        state_.cycle_count++;
        state_.last_cycle = cycle_start;
        cycle_count_snapshot = state_.cycle_count;
    }

    // Architecture-aware optimizations (no state_ access required).
    optimize_for_architecture();

    // Attention decay. atom_space has its own internal locking, so we only
    // need state_mutex_ to read cycle_count above; the decay call itself
    // does not need our lock.
    if (cycle_count_snapshot % 10 == 0) {  // Every 10 cycles
        state_.atomspace->decay_attention();
    }

    auto cycle_end = std::chrono::steady_clock::now();
    auto cycle_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cycle_end - cycle_start);

    // Maintain cycle frequency. Sleep without holding state_mutex_.
    auto target_cycle_time = std::chrono::milliseconds(static_cast<int>(1000.0 / cycle_frequency_hz_));
    if (cycle_duration < target_cycle_time) {
        std::this_thread::sleep_for(target_cycle_time - cycle_duration);
    }
}

void cognitive_cycle_manager::run_continuous(size_t max_cycles) {
    running_ = true;
    size_t cycles_run = 0;

    while (running_ && (max_cycles == 0 || cycles_run < max_cycles)) {
        run_single_cycle();
        cycles_run++;
    }
}

void cognitive_cycle_manager::stop() {
    running_ = false;
}

void cognitive_cycle_manager::add_goal(const goal_t& goal) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.active_goals.push(goal);
}

void cognitive_cycle_manager::remove_goal(const std::string& description) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    // Rebuild priority queue without the specified goal
    std::priority_queue<goal_t> new_goals;
    auto temp_goals = state_.active_goals;

    while (!temp_goals.empty()) {
        goal_t g = temp_goals.top();
        temp_goals.pop();
        if (g.description != description) {
            new_goals.push(g);
        }
    }

    state_.active_goals = new_goals;
}

std::vector<goal_t> cognitive_cycle_manager::get_active_goals() const {
    std::lock_guard<std::mutex> lock(state_mutex_);

    std::vector<goal_t> goals;
    auto temp_goals = state_.active_goals;

    while (!temp_goals.empty()) {
        goals.push_back(temp_goals.top());
        temp_goals.pop();
    }

    return goals;
}

cognitive_state cognitive_cycle_manager::get_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
}

void cognitive_cycle_manager::set_architecture_config(const architecture_config& config) {
    arch_config_ = config;
    llm_engine_->set_architecture_optimization(config);
}

const architecture_config& cognitive_cycle_manager::get_architecture_config() const {
    return arch_config_;
}

void cognitive_cycle_manager::process_input(const std::string& input) {
    // Create concept node for the input. atom_space operations are
    // internally synchronised, so they do not need state_mutex_.
    auto input_node = state_.atomspace->add_node(atom_type::CONCEPT_NODE, "input:" + input);
    input_node->set_attention_value(attention_value(0.9, 0.8));  // High importance and urgency

    // Mutate state_ under state_mutex_. Inline the goal push instead of
    // calling add_goal() so we do not attempt to re-acquire the same
    // (non-recursive) mutex.
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.current_context = input;
    state_.active_goals.push(goal_t("Process input: " + input, 0.8));
}

std::string cognitive_cycle_manager::generate_response(const std::string& context) {
    if (!llm_engine_->is_model_loaded()) {
        return "Error: No language model loaded";
    }

    // Snapshot the parts of state_ we need under the lock so the LLM call
    // (which can take a while) does not block the cognitive worker thread
    // and is not affected by concurrent mutations.
    std::string effective_context;
    cognitive_state state_snapshot(state_.atomspace);
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        effective_context = context.empty() ? state_.current_context : context;
        state_snapshot = state_;
    }

    // Get relevant knowledge from atomspace (atomspace has its own locking).
    auto relevant_atoms = select_relevant_knowledge(effective_context);
    (void)relevant_atoms;

    // Generate response using cognitive inference
    return llm_engine_->cognitive_inference(effective_context,
                                          state_snapshot, arch_config_);
}

void cognitive_cycle_manager::set_cycle_frequency_hz(double frequency) {
    cycle_frequency_hz_ = std::max(0.1, std::min(100.0, frequency));
}

void cognitive_cycle_manager::set_attention_decay_rate(double rate) {
    attention_decay_rate_ = std::max(0.0, std::min(1.0, rate));
}

// Private methods
void cognitive_cycle_manager::perception_phase() {
    // Update attentional focus
    update_attentional_focus();

    // Process sensory inputs (if any)
    // This would typically involve processing external stimuli
}

void cognitive_cycle_manager::goal_selection_phase() {
    // Select highest priority goal for processing
    if (!state_.active_goals.empty()) {
        const goal_t& current_goal = state_.active_goals.top();

        // Ensure a corresponding goal atom exists in the atom_space.
        // add_node is idempotent (it returns the existing atom for the same
        // type+name), so this is safe to call on every cycle without
        // creating duplicates. We avoid mutating a local copy of the queued
        // goal because std::priority_queue exposes only a const top().
        auto goal_node = state_.atomspace->add_node(
            atom_type::CONCEPT_NODE, "goal:" + current_goal.description);
        goal_node->set_attention_value(attention_value(current_goal.priority, 0.7));
    }
}

void cognitive_cycle_manager::action_selection_phase() {
    // Select actions based on current goals and context
    if (!state_.active_goals.empty()) {
        // This is where we would select appropriate actions
        // For now, we'll just mark high-attention atoms for processing
        auto focus_atoms = state_.atomspace->get_attentional_focus(20);
        state_.current_focus = focus_atoms;
    }
}

void cognitive_cycle_manager::execution_phase() {
    // Execute selected actions
    // This could involve calling external systems, updating the atomspace, etc.

    // Check for goal completion
    process_goal_completion();
}

void cognitive_cycle_manager::learning_phase() {
    // Update truth values based on recent experiences
    state_.atomspace->update_attention_values();

    // Consolidate knowledge (simplified)
    if (state_.cycle_count % 100 == 0) {  // Every 100 cycles
        // Perform more intensive learning/consolidation
        manage_memory_constraints();
    }
}

void cognitive_cycle_manager::optimize_for_architecture() {
    switch (arch_config_.type) {
        case architecture_type::CPU_ONLY:
            // Optimize for CPU processing
            break;
        case architecture_type::GPU_ACCELERATED:
            // Optimize for GPU acceleration
            break;
        case architecture_type::HYBRID_CPU_GPU:
            // Balance between CPU and GPU
            balance_inference_load();
            break;
        case architecture_type::MOBILE_OPTIMIZED:
            // Optimize for mobile constraints
            manage_memory_constraints();
            break;
        default:
            break;
    }
}

void cognitive_cycle_manager::manage_memory_constraints() {
    // If atomspace is getting too large, remove low-attention atoms
    if (state_.atomspace->size() > arch_config_.memory_limit_mb * 100) {  // Rough estimate
        // This is where we would implement memory management
        // For now, just decay attention more aggressively
        state_.atomspace->decay_attention();
    }
}

void cognitive_cycle_manager::balance_inference_load() {
    // Balance processing between different components
    // This would involve more sophisticated load balancing
}

void cognitive_cycle_manager::update_attentional_focus() {
    state_.current_focus = state_.atomspace->get_attentional_focus(50);
}

void cognitive_cycle_manager::process_goal_completion() {
    // Check if any goals have been completed
    auto temp_goals = state_.active_goals;
    std::priority_queue<goal_t> remaining_goals;

    while (!temp_goals.empty()) {
        goal_t g = temp_goals.top();
        temp_goals.pop();

        // Simple completion check - in reality this would be more sophisticated
        auto elapsed = std::chrono::steady_clock::now() - g.created;
        if (elapsed < std::chrono::minutes(5)) {  // Keep goals active for 5 minutes
            remaining_goals.push(g);
        }
    }

    state_.active_goals = remaining_goals;
}

std::vector<std::shared_ptr<Atom>> cognitive_cycle_manager::select_relevant_knowledge(const std::string& context) const {
    // Query atomspace for relevant atoms
    auto query_atoms = state_.atomspace->query(context);

    // Also include high-attention atoms
    auto focus_atoms = state_.atomspace->get_attentional_focus(20);

    // Combine and deduplicate
    std::vector<std::shared_ptr<Atom>> relevant;
    relevant.insert(relevant.end(), query_atoms.begin(), query_atoms.end());
    relevant.insert(relevant.end(), focus_atoms.begin(), focus_atoms.end());

    // Remove duplicates (simplified)
    std::sort(relevant.begin(), relevant.end());
    relevant.erase(std::unique(relevant.begin(), relevant.end()), relevant.end());

    return relevant;
}

// embodied_reasoning_engine implementation
embodied_reasoning_engine::embodied_reasoning_engine(std::shared_ptr<atom_space> atomspace)
    : atomspace_(atomspace), current_context_("default") {
}

void embodied_reasoning_engine::add_spatial_knowledge(const std::string& object, const std::string& location) {
    auto spatial_atom = create_spatial_atom(object, location);
    spatial_atom->set_truth_value(truth_value(0.9, 0.8));  // High confidence in spatial facts
}

void embodied_reasoning_engine::add_temporal_knowledge(const std::string& event, const std::string& time_context) {
    auto temporal_atom = create_temporal_atom(event, time_context);
    temporal_atom->set_truth_value(truth_value(0.8, 0.7));
}

void embodied_reasoning_engine::add_causal_relation(const std::string& cause, const std::string& effect, double confidence) {
    auto causal_atom = create_causal_atom(cause, effect);
    causal_atom->set_truth_value(truth_value(0.8, confidence));
}

std::vector<std::string> embodied_reasoning_engine::infer_consequences(const std::string& action) const {
    std::vector<std::string> consequences;

    // Query for causal relations involving this action
    auto causal_atoms = atomspace_->query(action);

    for (auto atom : causal_atoms) {
        if (atom->is_link() && atom->get_type() == atom_type::IMPLICATION_LINK) {
            // Extract consequence from implication link
            auto link = std::static_pointer_cast<Link>(atom);
            if (link->get_arity() >= 2) {
                auto effect = link->get_outgoing()[1];
                consequences.push_back(effect->to_string());
            }
        }
    }

    return consequences;
}

void embodied_reasoning_engine::update_context(const std::string& new_context) {
    current_context_ = new_context;

    // Create context atom
    auto context_node = atomspace_->add_node(atom_type::CONCEPT_NODE, "context:" + new_context);
    context_node->set_attention_value(attention_value(0.8, 0.6));
}

std::string embodied_reasoning_engine::get_current_context() const {
    return current_context_;
}

std::vector<std::string> embodied_reasoning_engine::plan_actions(const std::string& goal) const {
    std::vector<std::string> plan;

    // Simple planning based on goal decomposition
    // In a full implementation, this would use more sophisticated planning algorithms

    if (goal.find("move") != std::string::npos) {
        plan.push_back("check_current_location");
        plan.push_back("plan_path");
        plan.push_back("execute_movement");
        plan.push_back("verify_arrival");
    } else if (goal.find("learn") != std::string::npos) {
        plan.push_back("identify_knowledge_gap");
        plan.push_back("search_for_information");
        plan.push_back("integrate_new_knowledge");
        plan.push_back("validate_understanding");
    }

    return plan;
}

bool embodied_reasoning_engine::validate_action_feasibility(const std::string& action, const std::string& context) const {
    // Simple feasibility check
    // In reality, this would involve complex reasoning about physical constraints, capabilities, etc.

    auto context_atoms = atomspace_->query(context);
    auto action_atoms = atomspace_->query(action);

    // If we have knowledge about both context and action, assume feasible
    return !context_atoms.empty() && !action_atoms.empty();
}

// Private methods
std::shared_ptr<Atom> embodied_reasoning_engine::create_spatial_atom(const std::string& object, const std::string& location) {
    auto object_node = atomspace_->add_node(atom_type::CONCEPT_NODE, object);
    auto location_node = atomspace_->add_node(atom_type::CONCEPT_NODE, location);
    auto at_predicate = atomspace_->add_node(atom_type::PREDICATE_NODE, "at");

    // Create (EvaluationLink (PredicateNode "at") (ListLink object location))
    std::vector<std::shared_ptr<Atom>> args = {object_node, location_node};
    auto list_link = atomspace_->add_link(atom_type::INHERITANCE_LINK, args);  // Simplified

    std::vector<std::shared_ptr<Atom>> eval_args = {at_predicate, list_link};
    return atomspace_->add_link(atom_type::EVALUATION_LINK, eval_args);
}

std::shared_ptr<Atom> embodied_reasoning_engine::create_temporal_atom(const std::string& event, const std::string& time) {
    auto event_node = atomspace_->add_node(atom_type::CONCEPT_NODE, event);
    auto time_node = atomspace_->add_node(atom_type::CONCEPT_NODE, time);
    auto during_predicate = atomspace_->add_node(atom_type::PREDICATE_NODE, "during");

    std::vector<std::shared_ptr<Atom>> args = {event_node, time_node};
    auto list_link = atomspace_->add_link(atom_type::INHERITANCE_LINK, args);

    std::vector<std::shared_ptr<Atom>> eval_args = {during_predicate, list_link};
    return atomspace_->add_link(atom_type::EVALUATION_LINK, eval_args);
}

std::shared_ptr<Atom> embodied_reasoning_engine::create_causal_atom(const std::string& cause, const std::string& effect) {
    auto cause_node = atomspace_->add_node(atom_type::CONCEPT_NODE, cause);
    auto effect_node = atomspace_->add_node(atom_type::CONCEPT_NODE, effect);

    std::vector<std::shared_ptr<Atom>> args = {cause_node, effect_node};
    return atomspace_->add_link(atom_type::IMPLICATION_LINK, args);
}

} // namespace opencog
