#include "opencog/llm-inference.h"

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
        auto atomspace = std::make_shared<opencog::atom_space>();
        auto concept_node = atomspace->add_node(opencog::atom_type::CONCEPT_NODE, "robot");
        std::vector<std::shared_ptr<opencog::atom>> knowledge = {concept_node};

        opencog::cognitive_prompt_engine prompt_engine(atomspace);

        auto enhanced = prompt_engine.enhance_prompt("Where is the robot?", knowledge, "robot navigation");
        require_true(enhanced.find("Query: Where is the robot?") != std::string::npos, "enhanced prompt should include query");
        require_true(enhanced.find("Reasoning context: robot navigation") != std::string::npos, "enhanced prompt should include reasoning context");

        opencog::goal g("find charging station", 0.7);
        auto goal_prompt = prompt_engine.create_goal_prompt(g, knowledge);
        require_true(goal_prompt.find("Goal: find charging station") != std::string::npos, "goal prompt should include goal text");

        auto chain = prompt_engine.create_reasoning_chain("How to reach the charging station?", knowledge);
        require_true(chain.find("step-by-step logical reasoning") != std::string::npos, "reasoning chain should include reasoning template");

        auto embodied = prompt_engine.create_embodied_prompt("move_to_station", "factory_floor", knowledge);
        require_true(embodied.find("Action to perform: move_to_station") != std::string::npos, "embodied prompt should include action");
        require_true(embodied.find("Environment: factory_floor") != std::string::npos, "embodied prompt should include environment");

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "test-prompt-engine failed: " << e.what() << std::endl;
        return 1;
    }
}
