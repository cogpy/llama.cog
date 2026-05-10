#include "opencog/cognitive-cycle.h"
#include "opencog/llm-inference.h"

#include <iostream>
#include <memory>
#include <stdexcept>

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
        auto llm = std::make_shared<opencog::llm_inference_engine>();
        opencog::cognitive_cycle_manager cycle(atomspace, llm);

        constexpr double out_of_range_frequency = 120.0;
        constexpr double out_of_range_decay_rate = 2.0;
        cycle.set_cycle_frequency_hz(out_of_range_frequency);
        cycle.set_attention_decay_rate(out_of_range_decay_rate);
        cycle.add_goal(opencog::goal("high-priority-goal", 0.9));
        cycle.add_goal(opencog::goal("low-priority-goal", 0.2));

        auto goals = cycle.get_active_goals();
        require_true(!goals.empty(), "goals should be present after add_goal");
        require_true(goals.front().description == "high-priority-goal", "highest priority goal should be first");

        cycle.process_input("hello world");
        auto state_before = cycle.get_state();
        require_true(state_before.current_context == "hello world", "process_input should set context");

        cycle.run_single_cycle();
        auto state_after = cycle.get_state();
        require_true(state_after.cycle_count == state_before.cycle_count + 1, "single cycle should increment cycle count");
        require_true(!state_after.current_focus.empty(), "single cycle should update attentional focus");

        auto response = cycle.generate_response();
        require_true(response == "Error: No language model loaded", "without a model, generate_response should return an error");

        cycle.remove_goal("high-priority-goal");
        auto remaining_goals = cycle.get_active_goals();
        for (const auto & g : remaining_goals) {
            require_true(g.description != "high-priority-goal", "remove_goal should remove matching goals");
        }

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "test-cognitive-cycle failed: " << e.what() << std::endl;
        return 1;
    }
}
