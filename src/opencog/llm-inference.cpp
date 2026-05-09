#include "opencog/llm-inference.h"
#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <thread>

namespace opencog {

// llm_inference_engine implementation
llm_inference_engine::llm_inference_engine()
    : model_(nullptr), context_(nullptr), sampler_(nullptr), model_loaded_(false) {
}

llm_inference_engine::~llm_inference_engine() {
    unload_model();
}

bool llm_inference_engine::load_model(const std::string& model_path, const architecture_config& config) {
    if (model_loaded_) {
        unload_model();
    }

    current_config_ = config;

    // Initialize llama.cpp backend. ggml_backend_load_all() must be called
    // so the dynamic backends (CUDA, Vulkan, Metal, ...) are discovered;
    // without it, n_gpu_layers > 0 has no effect and inference silently
    // falls back to CPU. Backend init/free must be balanced exactly once;
    // backend_initialized_ guards against double-free if load_model fails
    // partway through and the destructor later runs unload_model.
    if (!backend_initialized_) {
        llama_backend_init();
        ggml_backend_load_all();
        backend_initialized_ = true;
    }

    // Set up model parameters based on architecture config
    llama_model_params model_params = llama_model_default_params();

    switch (config.type) {
        case architecture_type::GPU_ACCELERATED:
            model_params.n_gpu_layers = 99; // Use maximum GPU layers
            break;
        case architecture_type::HYBRID_CPU_GPU:
            model_params.n_gpu_layers = 20; // Use some GPU layers
            break;
        case architecture_type::MOBILE_OPTIMIZED:
            model_params.n_gpu_layers = 0;  // CPU only for mobile
            break;
        default:
            model_params.n_gpu_layers = 0;  // CPU only by default
            break;
    }

    // Load model
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        // Leave backend init in place; unload_model() / destructor will
        // free it exactly once via backend_initialized_.
        return false;
    }

    // Set up context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = std::min(config.max_context_length, static_cast<size_t>(4096));
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;

    // Create context
    context_ = llama_init_from_model(model_, ctx_params);
    if (!context_) {
        llama_model_free(model_);
        model_ = nullptr;
        // Leave backend init in place; unload_model() / destructor will
        // free it exactly once via backend_initialized_.
        return false;
    }

    // Create sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    model_loaded_ = true;

    // Apply architecture-specific optimizations
    set_architecture_optimization(config);

    return true;
}

void llm_inference_engine::unload_model() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }

    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }

    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }

    if (backend_initialized_) {
        llama_backend_free();
        backend_initialized_ = false;
    }
    model_loaded_ = false;
}

bool llm_inference_engine::is_model_loaded() const {
    return model_loaded_;
}

std::string llm_inference_engine::generate_response(const std::string& prompt,
                                                const std::vector<std::shared_ptr<atom>>& context_atoms,
                                                size_t max_tokens) {
    if (!model_loaded_) {
        return "Error: No model loaded";
    }

    auto start_time = std::chrono::steady_clock::now();

    // Format prompt with cognitive context
    std::string enhanced_prompt = prompt;
    if (!context_atoms.empty()) {
        enhanced_prompt = format_prompt_with_reasoning(prompt, context_atoms);
    }

    // Tokenize the prompt
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens;
    tokens.resize(enhanced_prompt.length() + 256);
    int n_tokens = llama_tokenize(vocab, enhanced_prompt.c_str(), enhanced_prompt.length(),
                                  tokens.data(), tokens.size(), true, false);

    if (n_tokens <= 0) {
        // n_tokens < 0  -> tokenize failed (insufficient buffer or other error).
        // n_tokens == 0 -> empty tokenization; passes the < 0 check but would
        //                  cause batch.logits[-1] OOB at the line below.
        return n_tokens < 0 ? "Error: Failed to tokenize prompt"
                            : "Error: Empty tokenization result";
    }

    tokens.resize(n_tokens);

    // Evaluate the prompt
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = n_tokens;
    batch.logits[batch.n_tokens - 1] = true; // Only calculate logits for the last token

    if (llama_decode(context_, batch) != 0) {
        llama_batch_free(batch);
        return "Error: Failed to decode prompt";
    }

    // Generate response
    std::string response;
    size_t tokens_generated = 0;

    for (size_t i = 0; i < max_tokens; ++i) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler_, context_, batch.n_tokens - 1);

        // Check for end of sequence
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Convert token to text
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        char token_str[256];
        int n_chars = llama_token_to_piece(vocab, new_token, token_str, sizeof(token_str), 0, false);
        if (n_chars > 0) {
            response.append(token_str, n_chars);
        }

        // Accept the token
        // llama_sampler_sample already calls llama_sampler_accept internally

        // Prepare for next iteration
        llama_batch_free(batch);
        batch = llama_batch_init(1, 0, 1);
        batch.token[0] = new_token;
        batch.pos[0] = n_tokens + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;

        if (llama_decode(context_, batch) != 0) {
            tokens_generated++;
            break;
        }

        tokens_generated++;
    }

    llama_batch_free(batch);

    // Update metrics
    update_metrics(start_time, tokens_generated);

    return response;
}

std::string llm_inference_engine::cognitive_inference(const std::string& input,
                                                  const cognitive_state& state,
                                                  const architecture_config& config) {
    if (!model_loaded_) {
        return "Error: No model loaded";
    }

    // Create cognitive prompt with reasoning context
    std::stringstream cognitive_prompt;
    cognitive_prompt << "You are an advanced cognitive AI system with access to a knowledge base.\n\n";

    // Add context from atomspace
    if (!state.current_focus.empty()) {
        cognitive_prompt << "Current knowledge context:\n";
        for (const auto& a : state.current_focus) {
            cognitive_prompt << "- " << a->to_string() << "\n";
        }
        cognitive_prompt << "\n";
    }

    // Add active goals
    if (!state.active_goals.empty()) {
        cognitive_prompt << "Active goals:\n";
        auto temp_goals = state.active_goals;
        int goal_count = 0;
        while (!temp_goals.empty() && goal_count < 3) {
            goal g = temp_goals.top();
            temp_goals.pop();
            cognitive_prompt << "- " << g.description << " (priority: " << g.priority << ")\n";
            goal_count++;
        }
        cognitive_prompt << "\n";
    }

    // Add the main input
    cognitive_prompt << "Input: " << input << "\n\n";
    cognitive_prompt << "Please provide a thoughtful response considering the knowledge context and goals:";

    // Generate response with architecture awareness
    size_t max_tokens = std::min(config.max_context_length / 4, static_cast<size_t>(512));
    return generate_response(cognitive_prompt.str(), state.current_focus, max_tokens);
}

std::string llm_inference_engine::reason_and_generate(const std::string& query,
                                                  const std::vector<std::shared_ptr<atom>>& knowledge_base) {
    std::stringstream reasoning_prompt;
    reasoning_prompt << "Given the following knowledge, reason step by step to answer the query.\n\n";

    reasoning_prompt << "Knowledge:\n";
    for (const auto& a : knowledge_base) {
        reasoning_prompt << "- " << a->to_string() << "\n";
    }

    reasoning_prompt << "\nQuery: " << query << "\n\n";
    reasoning_prompt << "Reasoning steps:\n1. ";

    return generate_response(reasoning_prompt.str(), knowledge_base, 256);
}

std::vector<float> llm_inference_engine::get_embeddings(const std::string& text) {
    if (!model_loaded_) {
        return {};
    }

    // Tokenize text
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens;
    tokens.resize(text.length() + 256);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(),
                                  tokens.data(), tokens.size(), true, false);

    if (n_tokens <= 0) {
        return {};
    }

    tokens.resize(n_tokens);

    // Get embeddings (simplified - in reality would use proper embedding extraction)
    const int embedding_size = llama_model_n_embd(model_);
    std::vector<float> embeddings(embedding_size, 0.0f);

    // This is a placeholder - real implementation would extract embeddings properly
    for (int i = 0; i < embedding_size && i < n_tokens; ++i) {
        embeddings[i] = static_cast<float>(tokens[i % n_tokens]) / 32000.0f;
    }

    return embeddings;
}

double llm_inference_engine::compute_semantic_similarity(const std::string& text1, const std::string& text2) {
    auto emb1 = get_embeddings(text1);
    auto emb2 = get_embeddings(text2);

    if (emb1.empty() || emb2.empty() || emb1.size() != emb2.size()) {
        return 0.0;
    }

    // Compute cosine similarity
    double dot_product = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;

    for (size_t i = 0; i < emb1.size(); ++i) {
        dot_product += static_cast<double>(emb1[i]) * static_cast<double>(emb2[i]);
        norm1 += static_cast<double>(emb1[i]) * static_cast<double>(emb1[i]);
        norm2 += static_cast<double>(emb2[i]) * static_cast<double>(emb2[i]);
    }

    if (norm1 == 0.0 || norm2 == 0.0) {
        return 0.0;
    }

    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

void llm_inference_engine::update_context(const std::vector<std::shared_ptr<atom>>& atoms) {
    // This would update the LLM context with relevant atoms
    // For now, we'll just store them for use in the next generation
    (void)atoms;
}

void llm_inference_engine::clear_context() {
    if (context_) {
        llama_memory_clear(llama_get_memory(context_), true);
    }
}

llm_inference_engine::inference_metrics llm_inference_engine::get_last_metrics() const {
    return last_metrics_;
}

void llm_inference_engine::set_sampling_params(float temperature, float top_p, int top_k) {
    if (sampler_) {
        llama_sampler_free(sampler_);
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

void llm_inference_engine::set_architecture_optimization(const architecture_config& config) {
    current_config_ = config;

    switch (config.type) {
        case architecture_type::CPU_ONLY:
            optimize_for_cpu();
            break;
        case architecture_type::GPU_ACCELERATED:
            optimize_for_gpu();
            break;
        case architecture_type::HYBRID_CPU_GPU:
            optimize_for_hybrid();
            break;
        case architecture_type::DISTRIBUTED:
            // Distributed optimizations - placeholder
            optimize_for_hybrid();  // Use hybrid as fallback
            break;
        case architecture_type::MOBILE_OPTIMIZED:
            optimize_for_mobile();
            break;
    }
}

// Private methods
std::string llm_inference_engine::atoms_to_context_string(const std::vector<std::shared_ptr<atom>>& atoms) const {
    std::stringstream ss;
    ss << "Knowledge context:\n";
    for (const auto& a : atoms) {
        ss << "- " << a->to_string() << "\n";
    }
    return ss.str();
}

std::string llm_inference_engine::format_prompt_with_reasoning(const std::string& prompt,
                                                           const std::vector<std::shared_ptr<atom>>& reasoning_atoms) const {
    std::stringstream formatted;
    formatted << atoms_to_context_string(reasoning_atoms);
    formatted << "\nPrompt: " << prompt << "\n\nResponse:";
    return formatted.str();
}

void llm_inference_engine::optimize_for_cpu() {
    // CPU-specific optimizations
    set_sampling_params(0.7f, 0.9f, 30);  // Faster sampling
}

void llm_inference_engine::optimize_for_gpu() {
    // GPU-specific optimizations
    set_sampling_params(0.8f, 0.95f, 50);  // Higher quality sampling
}

void llm_inference_engine::optimize_for_hybrid() {
    // Hybrid optimizations
    set_sampling_params(0.8f, 0.9f, 40);  // Balanced sampling
}

void llm_inference_engine::optimize_for_mobile() {
    // Mobile optimizations - prioritize speed and efficiency
    set_sampling_params(0.6f, 0.8f, 20);  // Fastest sampling
}

void llm_inference_engine::update_metrics(const std::chrono::steady_clock::time_point& start_time,
                                      size_t tokens_generated) {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    last_metrics_.processing_time_ms = duration.count();
    last_metrics_.tokens_generated = tokens_generated;
    last_metrics_.tokens_per_second = (duration.count() > 0) ? (tokens_generated * 1000.0 / duration.count()) : 0.0;
    last_metrics_.context_length = llama_n_ctx(context_);
    last_metrics_.memory_usage_mb = 0;  // Would need system-specific implementation
}

// cognitive_prompt_engine implementation
cognitive_prompt_engine::cognitive_prompt_engine(std::shared_ptr<atom_space> atomspace)
    : atomspace_(atomspace) {
}

std::string cognitive_prompt_engine::enhance_prompt(const std::string& base_prompt,
                                                const std::vector<std::shared_ptr<atom>>& relevant_atoms,
                                                const std::string& reasoning_context) const {
    std::stringstream enhanced;

    enhanced << "You are an advanced cognitive AI system. Use the following context to provide a thoughtful response.\n\n";

    if (!relevant_atoms.empty()) {
        enhanced << format_atoms_as_knowledge(relevant_atoms) << "\n";
    }

    if (!reasoning_context.empty()) {
        enhanced << "Reasoning context: " << reasoning_context << "\n\n";
    }

    enhanced << "Query: " << base_prompt << "\n\n";
    enhanced << "Please provide a response that demonstrates reasoning based on the given context:";

    return enhanced.str();
}

std::string cognitive_prompt_engine::create_goal_prompt(const goal& g,
                                                    const std::vector<std::shared_ptr<atom>>& knowledge) const {
    std::stringstream goal_prompt;

    goal_prompt << "You are working toward the following goal:\n";
    goal_prompt << "Goal: " << g.description << "\n";
    goal_prompt << "Priority: " << g.priority << "\n\n";

    if (!knowledge.empty()) {
        goal_prompt << "Available knowledge:\n";
        goal_prompt << format_atoms_as_knowledge(knowledge) << "\n";
    }

    goal_prompt << "Based on the goal and available knowledge, what steps should be taken next?";

    return goal_prompt.str();
}

std::string cognitive_prompt_engine::create_reasoning_chain(const std::string& problem,
                                                        const std::vector<std::shared_ptr<atom>>& facts) const {
    std::stringstream chain_prompt;

    chain_prompt << create_reasoning_template() << "\n\n";
    chain_prompt << "Problem: " << problem << "\n\n";

    if (!facts.empty()) {
        chain_prompt << "Given facts:\n";
        chain_prompt << format_atoms_as_knowledge(facts) << "\n";
    }

    chain_prompt << "Please solve this step by step:";

    return chain_prompt.str();
}

std::string cognitive_prompt_engine::create_embodied_prompt(const std::string& action,
                                                        const std::string& environment_context,
                                                        const std::vector<std::shared_ptr<atom>>& spatial_knowledge) const {
    std::stringstream embodied_prompt;

    embodied_prompt << create_embodied_template() << "\n\n";
    embodied_prompt << "Action to perform: " << action << "\n";
    embodied_prompt << "Environment: " << environment_context << "\n\n";

    if (!spatial_knowledge.empty()) {
        embodied_prompt << "Spatial knowledge:\n";
        embodied_prompt << format_atoms_as_knowledge(spatial_knowledge) << "\n";
    }

    embodied_prompt << "Please reason about how to perform this action in the given environment:";

    return embodied_prompt.str();
}

std::string cognitive_prompt_engine::format_atoms_as_knowledge(const std::vector<std::shared_ptr<atom>>& atoms) const {
    std::stringstream knowledge;
    for (const auto& a : atoms) {
        knowledge << "- " << a->to_string() << "\n";
    }
    return knowledge.str();
}

std::string cognitive_prompt_engine::create_reasoning_template() const {
    return "You are an AI system capable of step-by-step logical reasoning. "
           "For each problem, break down your thinking into clear steps.";
}

std::string cognitive_prompt_engine::create_embodied_template() const {
    return "You are an embodied AI system that can reason about physical actions and spatial relationships. "
           "Consider the physical constraints and possibilities when planning actions.";
}

// cognitive_llm_memory implementation
cognitive_llm_memory::cognitive_llm_memory(std::shared_ptr<atom_space> atomspace)
    : atomspace_(atomspace) {
}

void cognitive_llm_memory::encode_interaction(const std::string& input, const std::string& output) {
    auto interaction_atom = create_interaction_atom(input, output);
    interaction_atom->set_truth_value(truth_value(0.8, 0.9));  // High confidence in recorded interactions
    interaction_atom->set_attention_value(attention_value(0.7, 0.5));
}

void cognitive_llm_memory::encode_reasoning_step(const std::string& premise, const std::string& conclusion) {
    auto reasoning_atom = create_reasoning_atom(premise, conclusion);
    reasoning_atom->set_truth_value(truth_value(0.9, 0.8));  // High confidence in reasoning steps
    reasoning_atom->set_attention_value(attention_value(0.8, 0.6));
}

std::vector<std::shared_ptr<atom>> cognitive_llm_memory::retrieve_relevant_memories(const std::string& query,
                                                                                size_t max_memories) const {
    auto all_memories = atomspace_->query(query);

    // Sort by relevance score
    std::sort(all_memories.begin(), all_memories.end(),
              [this, &query](const std::shared_ptr<atom>& a, const std::shared_ptr<atom>& b) {
                  return compute_relevance_score(a, query) > compute_relevance_score(b, query);
              });

    if (all_memories.size() > max_memories) {
        all_memories.resize(max_memories);
    }

    return all_memories;
}

void cognitive_llm_memory::learn_from_feedback(const std::string& interaction, bool positive_feedback) {
    auto memories = atomspace_->query(interaction);

    for (auto memory : memories) {
        truth_value tv = memory->get_truth_value();
        if (positive_feedback) {
            tv.strength = std::min(1.0, tv.strength + 0.1);
            tv.confidence = std::min(1.0, tv.confidence + 0.05);
        } else {
            tv.strength = std::max(0.0, tv.strength - 0.1);
            tv.confidence = std::max(0.0, tv.confidence - 0.05);
        }
        memory->set_truth_value(tv);
    }
}

void cognitive_llm_memory::update_concept_strengths(const std::vector<std::string>& concepts, double delta) {
    for (const std::string& concept_name : concepts) {
        auto concept_atoms = atomspace_->get_atoms_by_name(concept_name);
        for (auto a : concept_atoms) {
            truth_value tv = a->get_truth_value();
            tv.strength = std::max(0.0, std::min(1.0, tv.strength + delta));
            a->set_truth_value(tv);
        }
    }
}

void cognitive_llm_memory::consolidate_memories() {
    // Identify similar memories and merge them
    auto all_atoms = atomspace_->get_attentional_focus(1000);

    // Simple consolidation - boost attention of frequently accessed concepts
    for (auto a : all_atoms) {
        attention_value av = a->get_attention_value();
        av.importance *= 1.01;  // Small boost for active memories
        a->set_attention_value(av);
    }
}

void cognitive_llm_memory::decay_old_memories(double decay_rate) {
    // The decay magnitude is currently hard-coded inside
    // atom_space::decay_attention(); the rate parameter is reserved for a
    // future implementation that scales the decay per call.
    (void)decay_rate;
    atomspace_->decay_attention();
}

std::shared_ptr<atom> cognitive_llm_memory::create_interaction_atom(const std::string& input, const std::string& output) {
    auto input_node = atomspace_->add_node(atom_type::CONCEPT_NODE, "input:" + input);
    auto output_node = atomspace_->add_node(atom_type::CONCEPT_NODE, "output:" + output);
    auto interaction_predicate = atomspace_->add_node(atom_type::PREDICATE_NODE, "interaction");

    std::vector<std::shared_ptr<atom>> args = {input_node, output_node};
    auto list_link = atomspace_->add_link(atom_type::INHERITANCE_LINK, args);

    std::vector<std::shared_ptr<atom>> eval_args = {interaction_predicate, list_link};
    return atomspace_->add_link(atom_type::EVALUATION_LINK, eval_args);
}

std::shared_ptr<atom> cognitive_llm_memory::create_reasoning_atom(const std::string& premise, const std::string& conclusion) {
    auto premise_node = atomspace_->add_node(atom_type::CONCEPT_NODE, premise);
    auto conclusion_node = atomspace_->add_node(atom_type::CONCEPT_NODE, conclusion);

    std::vector<std::shared_ptr<atom>> args = {premise_node, conclusion_node};
    return atomspace_->add_link(atom_type::IMPLICATION_LINK, args);
}

double cognitive_llm_memory::compute_relevance_score(std::shared_ptr<atom> memory, const std::string& query) const {
    // Simple relevance scoring based on string similarity and attention
    std::string memory_str = memory->to_string();

    // Count common words (simplified)
    size_t common_words = 0;
    std::stringstream query_stream(query);
    std::string word;
    while (query_stream >> word) {
        if (memory_str.find(word) != std::string::npos) {
            common_words++;
        }
    }

    double text_similarity = static_cast<double>(common_words) / 10.0;  // Normalize
    double attention_score = memory->get_attention_value().importance;

    return 0.7 * text_similarity + 0.3 * attention_score;
}

} // namespace opencog
