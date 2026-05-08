#include "opencog/opencog.h"
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace opencog {

// cognitive_system implementation
cognitive_system::cognitive_system() : initialized_(false) {
    atomspace_ = std::make_shared<atom_space>();
    llm_engine_ = std::make_shared<llm_inference_engine>();
    reasoning_engine_ = std::make_shared<embodied_reasoning_engine>(atomspace_);
    prompt_engine_ = std::make_shared<cognitive_prompt_engine>(atomspace_);
    memory_manager_ = std::make_shared<cognitive_llm_memory>(atomspace_);
}

cognitive_system::~cognitive_system() {
    // Ensure the worker thread is stopped before the underlying resources
    // (atomspace_, llm_engine_, cognitive_cycle_) are destroyed. Without this
    // a still-joinable cycle_thread_ would trigger std::terminate when its
    // destructor runs.
    if (initialized_) {
        shutdown();
    } else if (cycle_thread_.joinable()) {
        if (cognitive_cycle_) {
            cognitive_cycle_->stop();
        }
        cycle_thread_.join();
    }
}

bool cognitive_system::initialize(const std::string& model_path, const architecture_config& config) {
    if (initialized_) {
        return true;
    }

    current_config_ = config;

    // Load LLM model
    if (!llm_engine_->load_model(model_path, config)) {
        return false;
    }

    // Initialize cognitive cycle after LLM is loaded
    cognitive_cycle_ = std::make_shared<cognitive_cycle_manager>(atomspace_, llm_engine_);
    cognitive_cycle_->set_architecture_config(config);

    // Mark the system as initialized before populating the default knowledge
    // base. The knowledge-loading helpers (add_knowledge / add_spatial_knowledge
    // / add_causal_knowledge) early-return when initialized_ is false, so they
    // would otherwise be silent no-ops here.
    initialized_ = true;

    // Setup default knowledge base
    setup_default_knowledge();

    return true;
}

void cognitive_system::shutdown() {
    if (!initialized_) {
        return;
    }

    // Stop the cognitive cycle and join the worker thread before tearing
    // down resources it may still be touching (atomspace_, llm_engine_).
    stop_continuous_processing();

    if (llm_engine_) {
        llm_engine_->unload_model();
    }

    if (atomspace_) {
        atomspace_->clear();
    }

    initialized_ = false;
}

bool cognitive_system::is_initialized() const {
    return initialized_;
}

std::string cognitive_system::process_query(const std::string& query) {
    if (!initialized_) {
        return "Error: System not initialized";
    }

    // Encode the interaction for learning
    memory_manager_->encode_interaction(query, "");

    // Process input through cognitive cycle
    cognitive_cycle_->process_input(query);

    // Generate enhanced response
    auto relevant_atoms = atomspace_->query(query);
    std::string enhanced_prompt = prompt_engine_->enhance_prompt(query, relevant_atoms);

    std::string response = cognitive_cycle_->generate_response(enhanced_prompt);

    // Learn from the interaction
    memory_manager_->encode_interaction(query, response);

    return response;
}

void cognitive_system::add_knowledge(const std::string& concept_name, const std::string& relation, const std::string& target) {
    if (!initialized_) {
        return;
    }

    auto concept_node = atomspace_->add_node(atom_type::CONCEPT_NODE, concept_name);
    concept_node->set_truth_value(truth_value(0.9, 0.8));

    if (!relation.empty() && !target.empty()) {
        auto target_node = atomspace_->add_node(atom_type::CONCEPT_NODE, target);
        auto relation_node = atomspace_->add_node(atom_type::PREDICATE_NODE, relation);

        std::vector<std::shared_ptr<Atom>> args = {concept_node, target_node};
        auto list_link = atomspace_->add_link(atom_type::INHERITANCE_LINK, args);

        std::vector<std::shared_ptr<Atom>> eval_args = {relation_node, list_link};
        auto evaluation = atomspace_->add_link(atom_type::EVALUATION_LINK, eval_args);
        evaluation->set_truth_value(truth_value(0.8, 0.7));
    }
}

void cognitive_system::set_goal(const std::string& goal_description, double priority) {
    if (!initialized_) {
        return;
    }

    goal_t goal(goal_description, priority);
    cognitive_cycle_->add_goal(goal);
}

void cognitive_system::configure_architecture(const architecture_config& config) {
    current_config_ = config;

    if (initialized_ && cognitive_cycle_) {
        cognitive_cycle_->set_architecture_config(config);
    }

    if (initialized_ && llm_engine_) {
        llm_engine_->set_architecture_optimization(config);
    }
}

void cognitive_system::optimize_for_hardware() {
    auto optimal_config = utils::create_optimal_config();
    configure_architecture(optimal_config);
}

void cognitive_system::add_spatial_knowledge(const std::string& object, const std::string& location) {
    if (!initialized_) {
        return;
    }

    reasoning_engine_->add_spatial_knowledge(object, location);
}

void cognitive_system::add_causal_knowledge(const std::string& cause, const std::string& effect, double confidence) {
    if (!initialized_) {
        return;
    }

    reasoning_engine_->add_causal_relation(cause, effect, confidence);
}

std::vector<std::string> cognitive_system::plan_actions(const std::string& goal) const {
    if (!initialized_) {
        return {};
    }

    return reasoning_engine_->plan_actions(goal);
}

cognitive_system::system_metrics cognitive_system::get_metrics() const {
    system_metrics metrics = {};

    if (initialized_) {
        metrics.atomspace_size = atomspace_->size();
        metrics.active_goals_count = cognitive_cycle_->get_active_goals().size();
        metrics.cycle_count = cognitive_cycle_->get_state().cycle_count;

        auto llm_metrics = llm_engine_->get_last_metrics();
        metrics.last_inference_time_ms = llm_metrics.processing_time_ms;
        metrics.tokens_per_second = llm_metrics.tokens_per_second;
        metrics.memory_usage_mb = llm_metrics.memory_usage_mb;
    }

    return metrics;
}

void cognitive_system::start_continuous_processing() {
    if (!initialized_) {
        return;
    }

    // If a previous cycle thread is still running, stop and join it first so
    // we never have more than one worker active at a time.
    if (cycle_thread_.joinable()) {
        if (cognitive_cycle_) {
            cognitive_cycle_->stop();
        }
        cycle_thread_.join();
    }

    // Start cognitive cycle in a separate, joinable thread. Keeping the
    // thread handle as a member lets shutdown() join it deterministically
    // before tearing down the atomspace and LLM engine, which avoids the
    // use-after-free that detached threads would otherwise allow.
    cycle_thread_ = std::thread([this]() {
        cognitive_cycle_->run_continuous();
    });
}

void cognitive_system::stop_continuous_processing() {
    if (cognitive_cycle_) {
        cognitive_cycle_->stop();
    }

    if (cycle_thread_.joinable()) {
        cycle_thread_.join();
    }
}

void cognitive_system::save_knowledge(const std::string& filename) const {
    if (!initialized_) {
        return;
    }

    std::ofstream file(filename);
    if (file.is_open()) {
        file << atomspace_->to_string();
        file.close();
    }
}

bool cognitive_system::load_knowledge(const std::string& filename) {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    // Parse lines matching the format produced by atom_space::to_string():
    //   "  (<type_id> \"<name>\") (TV: ...)"
    // Currently we only restore CONCEPT_NODE atoms (matched by their numeric
    // type id), which keeps load_knowledge consistent with save_knowledge.
    const std::string concept_prefix =
        "(" + std::to_string(static_cast<int>(atom_type::CONCEPT_NODE)) + " \"";
    std::string line;
    while (std::getline(file, line)) {
        size_t prefix_pos = line.find(concept_prefix);
        if (prefix_pos == std::string::npos) {
            continue;
        }

        // Compute the start of the name only after we've confirmed the
        // opening quote exists; otherwise unsigned overflow on `npos + 1`
        // would silently wrap to 0 and bypass any later guard.
        size_t name_start = prefix_pos + concept_prefix.size();
        size_t name_end = line.find('"', name_start);
        if (name_end == std::string::npos) {
            continue;
        }

        std::string concept_name = line.substr(name_start, name_end - name_start);
        if (!concept_name.empty()) {
            add_knowledge(concept_name);
        }
    }

    file.close();
    return true;
}

void cognitive_system::setup_default_knowledge() {
    // Add some basic cognitive concepts
    add_knowledge("artificial_intelligence", "is_a", "technology");
    add_knowledge("reasoning", "is_type_of", "cognitive_process");
    add_knowledge("language_model", "performs", "text_generation");
    add_knowledge("knowledge", "stored_in", "atomspace");

    // Add basic spatial concepts
    add_spatial_knowledge("AI_system", "digital_environment");

    // Add basic causal relations
    add_causal_knowledge("user_query", "system_response", 0.9);
    add_causal_knowledge("learning", "improved_performance", 0.8);
}

void cognitive_system::initialize_cognitive_components() {
    // Set default goals
    set_goal("Understand user queries", 0.8);
    set_goal("Provide helpful responses", 0.9);
    set_goal("Learn from interactions", 0.7);

    // Configure reasoning engine
    reasoning_engine_->update_context("interactive_session");
}

// Utility functions implementation
namespace utils {

architecture_type detect_optimal_architecture() {
    // Simple architecture detection
    #if defined(GGML_USE_CUDA) || defined(GGML_USE_VULKAN)
        return architecture_type::GPU_ACCELERATED;
    #elif defined(__APPLE__) && defined(__arm64__)
        return architecture_type::HYBRID_CPU_GPU;  // Apple Silicon
    #elif defined(__ANDROID__)
        return architecture_type::MOBILE_OPTIMIZED;
    #else
        return architecture_type::CPU_ONLY;
    #endif
}

architecture_config create_optimal_config() {
    architecture_config config;
    config.type = detect_optimal_architecture();

    switch (config.type) {
        case architecture_type::GPU_ACCELERATED:
            config.memory_limit_mb = 8192;
            config.max_context_length = 4096;
            config.inference_budget_ms = 200.0;
            config.enable_speculative_decoding = true;
            break;

        case architecture_type::HYBRID_CPU_GPU:
            config.memory_limit_mb = 6144;
            config.max_context_length = 3072;
            config.inference_budget_ms = 150.0;
            config.enable_speculative_decoding = true;
            break;

        case architecture_type::MOBILE_OPTIMIZED:
            config.memory_limit_mb = 2048;
            config.max_context_length = 1024;
            config.inference_budget_ms = 500.0;
            config.enable_speculative_decoding = false;
            break;

        default: // CPU_ONLY
            config.memory_limit_mb = 4096;
            config.max_context_length = 2048;
            config.inference_budget_ms = 300.0;
            config.enable_speculative_decoding = false;
            break;
    }

    return config;
}

std::vector<std::shared_ptr<Atom>> parse_knowledge_from_text(const std::string& text, atom_space& atomspace) {
    std::vector<std::shared_ptr<Atom>> atoms;

    // Simple knowledge extraction - look for concepts in text
    std::stringstream ss(text);
    std::string word;

    while (ss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(),
                                 [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }), word.end());

        if (word.length() > 3) {  // Only consider meaningful words
            auto atom = atomspace.add_node(atom_type::CONCEPT_NODE, word);
            atoms.push_back(atom);
        }
    }

    return atoms;
}

std::string atoms_to_readable_text(const std::vector<std::shared_ptr<Atom>>& atoms) {
    std::stringstream readable;

    for (size_t i = 0; i < atoms.size(); ++i) {
        if (atoms[i]->is_node()) {
            auto node = std::static_pointer_cast<Node>(atoms[i]);
            readable << node->get_name();
            if (i < atoms.size() - 1) {
                readable << ", ";
            }
        }
    }

    return readable.str();
}

void benchmark_inference_performance(llm_inference_engine& engine, const std::string& test_prompt) {
    if (!engine.is_model_loaded()) {
        return;
    }

    auto start = std::chrono::steady_clock::now();
    std::string response = engine.generate_response(test_prompt, {}, 100);
    auto end = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto metrics = engine.get_last_metrics();

    std::cout << "Benchmark Results:\n";
    std::cout << "  Total time: " << duration.count() << " ms\n";
    std::cout << "  Tokens/sec: " << metrics.tokens_per_second << "\n";
    std::cout << "  Response length: " << response.length() << " chars\n";
}

void profile_cognitive_cycle(cognitive_cycle_manager& cycle_manager, size_t num_cycles) {
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < num_cycles; ++i) {
        cycle_manager.run_single_cycle();
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Cognitive Cycle Profile:\n";
    std::cout << "  Cycles: " << num_cycles << "\n";
    std::cout << "  Total time: " << duration.count() << " ms\n";
    std::cout << "  Avg time per cycle: " << (duration.count() / num_cycles) << " ms\n";
}

} // namespace utils

} // namespace opencog
