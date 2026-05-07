#include "opencog/opencog.h"
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace opencog {

// CognitiveSystem implementation
CognitiveSystem::CognitiveSystem() : initialized_(false) {
    atomspace_ = std::make_shared<AtomSpace>();
    llm_engine_ = std::make_shared<LLMInferenceEngine>();
    reasoning_engine_ = std::make_shared<EmbodiedReasoningEngine>(atomspace_);
    prompt_engine_ = std::make_shared<CognitivePromptEngine>(atomspace_);
    memory_manager_ = std::make_shared<CognitiveLLMMemory>(atomspace_);
}

bool CognitiveSystem::initialize(const std::string& model_path, const ArchitectureConfig& config) {
    if (initialized_) {
        return true;
    }
    
    current_config_ = config;
    
    // Load LLM model
    if (!llm_engine_->load_model(model_path, config)) {
        return false;
    }
    
    // Initialize cognitive cycle after LLM is loaded
    cognitive_cycle_ = std::make_shared<CognitiveCycleManager>(atomspace_, llm_engine_);
    cognitive_cycle_->set_architecture_config(config);
    
    // Setup default knowledge base
    setup_default_knowledge();
    
    initialized_ = true;
    return true;
}

void CognitiveSystem::shutdown() {
    if (!initialized_) {
        return;
    }
    
    stop_continuous_processing();
    
    if (llm_engine_) {
        llm_engine_->unload_model();
    }
    
    if (atomspace_) {
        atomspace_->clear();
    }
    
    initialized_ = false;
}

bool CognitiveSystem::is_initialized() const {
    return initialized_;
}

std::string CognitiveSystem::process_query(const std::string& query) {
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

void CognitiveSystem::add_knowledge(const std::string& concept_name, const std::string& relation, const std::string& target) {
    if (!initialized_) {
        return;
    }
    
    auto concept_node = atomspace_->add_node(AtomType::CONCEPT_NODE, concept_name);
    concept_node->set_truth_value(TruthValue(0.9, 0.8));
    
    if (!relation.empty() && !target.empty()) {
        auto target_node = atomspace_->add_node(AtomType::CONCEPT_NODE, target);
        auto relation_node = atomspace_->add_node(AtomType::PREDICATE_NODE, relation);
        
        std::vector<std::shared_ptr<Atom>> args = {concept_node, target_node};
        auto list_link = atomspace_->add_link(AtomType::INHERITANCE_LINK, args);
        
        std::vector<std::shared_ptr<Atom>> eval_args = {relation_node, list_link};
        auto evaluation = atomspace_->add_link(AtomType::EVALUATION_LINK, eval_args);
        evaluation->set_truth_value(TruthValue(0.8, 0.7));
    }
}

void CognitiveSystem::set_goal(const std::string& goal_description, double priority) {
    if (!initialized_) {
        return;
    }
    
    Goal goal(goal_description, priority);
    cognitive_cycle_->add_goal(goal);
}

void CognitiveSystem::configure_architecture(const ArchitectureConfig& config) {
    current_config_ = config;
    
    if (initialized_ && cognitive_cycle_) {
        cognitive_cycle_->set_architecture_config(config);
    }
    
    if (initialized_ && llm_engine_) {
        llm_engine_->set_architecture_optimization(config);
    }
}

void CognitiveSystem::optimize_for_hardware() {
    auto optimal_config = utils::create_optimal_config();
    configure_architecture(optimal_config);
}

void CognitiveSystem::add_spatial_knowledge(const std::string& object, const std::string& location) {
    if (!initialized_) {
        return;
    }
    
    reasoning_engine_->add_spatial_knowledge(object, location);
}

void CognitiveSystem::add_causal_knowledge(const std::string& cause, const std::string& effect, double confidence) {
    if (!initialized_) {
        return;
    }
    
    reasoning_engine_->add_causal_relation(cause, effect, confidence);
}

std::vector<std::string> CognitiveSystem::plan_actions(const std::string& goal) const {
    if (!initialized_) {
        return {};
    }
    
    return reasoning_engine_->plan_actions(goal);
}

CognitiveSystem::SystemMetrics CognitiveSystem::get_metrics() const {
    SystemMetrics metrics = {};
    
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

void CognitiveSystem::start_continuous_processing() {
    if (!initialized_) {
        return;
    }
    
    // Start cognitive cycle in a separate thread
    std::thread cycle_thread([this]() {
        cognitive_cycle_->run_continuous();
    });
    cycle_thread.detach();
}

void CognitiveSystem::stop_continuous_processing() {
    if (initialized_ && cognitive_cycle_) {
        cognitive_cycle_->stop();
    }
}

void CognitiveSystem::save_knowledge(const std::string& filename) const {
    if (!initialized_) {
        return;
    }
    
    std::ofstream file(filename);
    if (file.is_open()) {
        file << atomspace_->to_string();
        file.close();
    }
}

bool CognitiveSystem::load_knowledge(const std::string& filename) {
    if (!initialized_) {
        return false;
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Simple knowledge loading - in reality would need proper parser
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("ConceptNode") != std::string::npos) {
            // Extract concept name and add to atomspace
            size_t start = line.find("\"") + 1;
            size_t end = line.find("\"", start);
            if (start != std::string::npos && end != std::string::npos) {
                std::string concept_name = line.substr(start, end - start);
                add_knowledge(concept_name);
            }
        }
    }
    
    file.close();
    return true;
}

void CognitiveSystem::setup_default_knowledge() {
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

void CognitiveSystem::initialize_cognitive_components() {
    // Set default goals
    set_goal("Understand user queries", 0.8);
    set_goal("Provide helpful responses", 0.9);
    set_goal("Learn from interactions", 0.7);
    
    // Configure reasoning engine
    reasoning_engine_->update_context("interactive_session");
}

// Utility functions implementation
namespace utils {

ArchitectureType detect_optimal_architecture() {
    // Simple architecture detection
    #if defined(GGML_USE_CUDA) || defined(GGML_USE_VULKAN)
        return ArchitectureType::GPU_ACCELERATED;
    #elif defined(__APPLE__) && defined(__arm64__)
        return ArchitectureType::HYBRID_CPU_GPU;  // Apple Silicon
    #elif defined(__ANDROID__)
        return ArchitectureType::MOBILE_OPTIMIZED;
    #else
        return ArchitectureType::CPU_ONLY;
    #endif
}

ArchitectureConfig create_optimal_config() {
    ArchitectureConfig config;
    config.type = detect_optimal_architecture();
    
    switch (config.type) {
        case ArchitectureType::GPU_ACCELERATED:
            config.memory_limit_mb = 8192;
            config.max_context_length = 4096;
            config.inference_budget_ms = 200.0;
            config.enable_speculative_decoding = true;
            break;
            
        case ArchitectureType::HYBRID_CPU_GPU:
            config.memory_limit_mb = 6144;
            config.max_context_length = 3072;
            config.inference_budget_ms = 150.0;
            config.enable_speculative_decoding = true;
            break;
            
        case ArchitectureType::MOBILE_OPTIMIZED:
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

std::vector<std::shared_ptr<Atom>> parse_knowledge_from_text(const std::string& text, AtomSpace& atomspace) {
    std::vector<std::shared_ptr<Atom>> atoms;
    
    // Simple knowledge extraction - look for concepts in text
    std::stringstream ss(text);
    std::string word;
    
    while (ss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), 
                                 [](char c) { return !std::isalnum(c); }), word.end());
        
        if (word.length() > 3) {  // Only consider meaningful words
            auto atom = atomspace.add_node(AtomType::CONCEPT_NODE, word);
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

void benchmark_inference_performance(LLMInferenceEngine& engine, const std::string& test_prompt) {
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

void profile_cognitive_cycle(CognitiveCycleManager& cycle_manager, size_t num_cycles) {
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