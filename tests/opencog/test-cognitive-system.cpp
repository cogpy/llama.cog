#include "opencog/opencog.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require_true(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}

int main() {
    try {
        opencog::cognitive_system system;

        require_true(!system.is_initialized(), "system should start uninitialized");
        require_true(system.process_query("hello") == "Error: System not initialized", "uninitialized system should reject queries");

        system.add_knowledge("concept");
        system.add_spatial_knowledge("robot", "lab");
        system.add_causal_knowledge("cause", "effect", 0.8);
        system.set_goal("goal", 0.5);
        require_true(system.plan_actions("move robot").empty(), "uninitialized system should not plan actions");

        auto metrics = system.get_metrics();
        require_true(metrics.atomspace_size == 0, "uninitialized metrics should be zeroed");
        require_true(metrics.active_goals_count == 0, "uninitialized metrics should be zeroed");
        require_true(metrics.cycle_count == 0, "uninitialized metrics should be zeroed");

        opencog::atom_space parser_atomspace;
        auto atoms = opencog::utils::parse_knowledge_from_text("Robots learn quickly in labs", parser_atomspace);
        require_true(!atoms.empty(), "knowledge parsing should extract concept atoms");
        auto readable = opencog::utils::atoms_to_readable_text(atoms);
        require_true(!readable.empty(), "readable text conversion should return content");

        auto config = opencog::utils::create_optimal_config();
        require_true(config.memory_limit_mb > 0, "optimal config should have positive memory limit");
        require_true(config.max_context_length > 0, "optimal config should have positive context length");

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "test-cognitive-system failed: " << e.what() << std::endl;
        return 1;
    }
}
