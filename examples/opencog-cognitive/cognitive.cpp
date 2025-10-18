#include "opencog/opencog.h"
#include "arg.h"
#include "common.h"
#include <iostream>
#include <string>
#include <vector>

/**
 * OpenCog Cognitive LLM Example
 * 
 * This example demonstrates the OpenCog-inspired cognitive architecture
 * integrated with llama.cpp for embodied, architecture-aware LLM inference.
 * 
 * Features demonstrated:
 * - AtomSpace knowledge representation
 * - Goal-driven cognitive processing
 * - Architecture-aware inference optimization
 * - Embodied reasoning capabilities
 * - Memory and learning from interactions
 */

void print_banner() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  OpenCog Cognitive LLM                       ║\n";
    std::cout << "║          Architecture-Aware Embodied AI System              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void print_help() {
    std::cout << "Commands:\n";
    std::cout << "  help          - Show this help message\n";
    std::cout << "  status        - Show system status and metrics\n";
    std::cout << "  goals         - List active cognitive goals\n";
    std::cout << "  knowledge     - Show current knowledge base\n";
    std::cout << "  add <concept> - Add a concept to the knowledge base\n";
    std::cout << "  goal <desc>   - Set a new cognitive goal\n";
    std::cout << "  spatial <obj> <loc> - Add spatial knowledge\n";
    std::cout << "  causal <cause> <effect> - Add causal relationship\n";
    std::cout << "  plan <goal>   - Plan actions for a goal\n";
    std::cout << "  cycle         - Run a single cognitive cycle\n";
    std::cout << "  continuous    - Start/stop continuous processing\n";
    std::cout << "  save <file>   - Save knowledge to file\n";
    std::cout << "  load <file>   - Load knowledge from file\n";
    std::cout << "  exit, quit    - Exit the program\n";
    std::cout << "\n";
}

void print_system_status(const opencog::CognitiveSystem& system) {
    auto metrics = system.get_metrics();
    
    std::cout << "\n=== System Status ===\n";
    std::cout << "AtomSpace size: " << metrics.atomspace_size << " atoms\n";
    std::cout << "Active goals: " << metrics.active_goals_count << "\n";
    std::cout << "Cognitive cycles: " << metrics.cycle_count << "\n";
    std::cout << "Last inference: " << metrics.last_inference_time_ms << " ms\n";
    std::cout << "Performance: " << metrics.tokens_per_second << " tokens/sec\n";
    std::cout << "Memory usage: " << metrics.memory_usage_mb << " MB\n";
    std::cout << "\n";
}

void show_active_goals(const opencog::CognitiveSystem& system) {
    auto goals = system.get_cognitive_cycle()->get_active_goals();
    
    std::cout << "\n=== Active Goals ===\n";
    if (goals.empty()) {
        std::cout << "No active goals\n";
    } else {
        for (size_t i = 0; i < goals.size(); ++i) {
            std::cout << (i + 1) << ". " << goals[i].description 
                      << " (priority: " << goals[i].priority << ")\n";
        }
    }
    std::cout << "\n";
}

void show_knowledge_base(const opencog::CognitiveSystem& system) {
    auto atomspace = system.get_atomspace();
    
    std::cout << "\n=== Knowledge Base ===\n";
    std::cout << "Total atoms: " << atomspace->size() << "\n";
    std::cout << "Nodes: " << atomspace->get_num_nodes() << "\n";
    std::cout << "Links: " << atomspace->get_num_links() << "\n";
    
    // Show high-attention atoms
    auto focus_atoms = atomspace->get_attentional_focus(10);
    if (!focus_atoms.empty()) {
        std::cout << "\nHigh attention atoms:\n";
        for (const auto& atom : focus_atoms) {
            std::cout << "- " << atom->to_string() << "\n";
        }
    }
    std::cout << "\n";
}

int main(int argc, char ** argv) {
    common_params params;
    
    // Parse command line arguments
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN)) {
        return 1;
    }
    
    print_banner();
    
    // Initialize cognitive system
    std::cout << "Initializing OpenCog Cognitive System...\n";
    
    opencog::CognitiveSystem cognitive_system;
    
    // Configure architecture based on available hardware
    opencog::ArchitectureConfig config = opencog::utils::create_optimal_config();
    std::cout << "Detected architecture: ";
    switch (config.type) {
        case opencog::ArchitectureType::CPU_ONLY:
            std::cout << "CPU-only\n";
            break;
        case opencog::ArchitectureType::GPU_ACCELERATED:
            std::cout << "GPU-accelerated\n";
            break;
        case opencog::ArchitectureType::HYBRID_CPU_GPU:
            std::cout << "Hybrid CPU/GPU\n";
            break;
        case opencog::ArchitectureType::DISTRIBUTED:
            std::cout << "Distributed\n";
            break;
        case opencog::ArchitectureType::MOBILE_OPTIMIZED:
            std::cout << "Mobile-optimized\n";
            break;
    }
    
    // Load model
    if (params.model.path.empty()) {
        std::cout << "No model specified. Please use -m <model_path>\n";
        return 1;
    }
    
    std::cout << "Loading model: " << params.model.path << "\n";
    if (!cognitive_system.initialize(params.model.path, config)) {
        std::cout << "Failed to initialize cognitive system\n";
        return 1;
    }
    
    std::cout << "System initialized successfully!\n";
    std::cout << "Type 'help' for available commands.\n\n";
    
    // Add some initial knowledge
    cognitive_system.add_knowledge("artificial_intelligence", "is_a", "cognitive_technology");
    cognitive_system.add_knowledge("reasoning", "enables", "problem_solving");
    cognitive_system.add_spatial_knowledge("AI_system", "digital_environment");
    cognitive_system.add_causal_knowledge("learning", "improved_performance", 0.9);
    
    // Set initial goals
    cognitive_system.set_goal("Understand user queries", 0.9);
    cognitive_system.set_goal("Provide helpful responses", 0.8);
    cognitive_system.set_goal("Learn from interactions", 0.7);
    
    bool continuous_mode = false;
    std::string line;
    
    // Main interaction loop
    while (true) {
        std::cout << "opencog> ";
        if (!std::getline(std::cin, line)) {
            break;
        }
        
        if (line.empty()) continue;
        
        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.empty()) continue;
        
        const std::string& command = tokens[0];
        
        if (command == "help") {
            print_help();
        }
        else if (command == "status") {
            print_system_status(cognitive_system);
        }
        else if (command == "goals") {
            show_active_goals(cognitive_system);
        }
        else if (command == "knowledge") {
            show_knowledge_base(cognitive_system);
        }
        else if (command == "add" && tokens.size() >= 2) {
            std::string concept_name = tokens[1];
            cognitive_system.add_knowledge(concept_name);
            std::cout << "Added concept: " << concept_name << "\n";
        }
        else if (command == "goal" && tokens.size() >= 2) {
            std::string goal_desc;
            for (size_t i = 1; i < tokens.size(); ++i) {
                goal_desc += tokens[i];
                if (i < tokens.size() - 1) goal_desc += " ";
            }
            cognitive_system.set_goal(goal_desc, 0.8);
            std::cout << "Added goal: " << goal_desc << "\n";
        }
        else if (command == "spatial" && tokens.size() >= 3) {
            cognitive_system.add_spatial_knowledge(tokens[1], tokens[2]);
            std::cout << "Added spatial knowledge: " << tokens[1] << " at " << tokens[2] << "\n";
        }
        else if (command == "causal" && tokens.size() >= 3) {
            cognitive_system.add_causal_knowledge(tokens[1], tokens[2]);
            std::cout << "Added causal relation: " << tokens[1] << " -> " << tokens[2] << "\n";
        }
        else if (command == "plan" && tokens.size() >= 2) {
            std::string goal;
            for (size_t i = 1; i < tokens.size(); ++i) {
                goal += tokens[i];
                if (i < tokens.size() - 1) goal += " ";
            }
            auto actions = cognitive_system.plan_actions(goal);
            std::cout << "Action plan for '" << goal << "':\n";
            for (size_t i = 0; i < actions.size(); ++i) {
                std::cout << (i + 1) << ". " << actions[i] << "\n";
            }
        }
        else if (command == "cycle") {
            std::cout << "Running cognitive cycle...\n";
            cognitive_system.get_cognitive_cycle()->run_single_cycle();
            std::cout << "Cycle completed.\n";
        }
        else if (command == "continuous") {
            if (continuous_mode) {
                std::cout << "Stopping continuous processing...\n";
                cognitive_system.stop_continuous_processing();
                continuous_mode = false;
            } else {
                std::cout << "Starting continuous processing...\n";
                cognitive_system.start_continuous_processing();
                continuous_mode = true;
            }
        }
        else if (command == "save" && tokens.size() >= 2) {
            cognitive_system.save_knowledge(tokens[1]);
            std::cout << "Knowledge saved to " << tokens[1] << "\n";
        }
        else if (command == "load" && tokens.size() >= 2) {
            if (cognitive_system.load_knowledge(tokens[1])) {
                std::cout << "Knowledge loaded from " << tokens[1] << "\n";
            } else {
                std::cout << "Failed to load knowledge from " << tokens[1] << "\n";
            }
        }
        else if (command == "exit" || command == "quit") {
            break;
        }
        else {
            // Treat as a query to the cognitive system
            std::cout << "\nProcessing query: " << line << "\n";
            std::cout << "Response: ";
            
            auto start = std::chrono::steady_clock::now();
            std::string response = cognitive_system.process_query(line);
            auto end = std::chrono::steady_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << response << "\n";
            std::cout << "\n[Processed in " << duration.count() << " ms]\n\n";
        }
    }
    
    std::cout << "\nShutting down cognitive system...\n";
    cognitive_system.shutdown();
    std::cout << "Goodbye!\n";
    
    return 0;
}