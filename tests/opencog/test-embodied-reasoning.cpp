#include "opencog/cognitive-cycle.h"

#include <iostream>
#include <algorithm>
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
        opencog::embodied_reasoning_engine reasoning(atomspace);

        reasoning.add_spatial_knowledge("robot", "lab");
        reasoning.add_temporal_knowledge("inspection", "morning");
        reasoning.add_causal_relation("move_robot", "robot_in_lab", 0.9);

        auto consequences = reasoning.infer_consequences("move_robot");
        require_true(!consequences.empty(), "causal relation should produce consequences");

        auto plan = reasoning.plan_actions("move robot to lab");
        require_true(!plan.empty(), "move goal should produce a non-empty movement plan");
        require_true(std::find(plan.begin(), plan.end(), "plan_path") != plan.end(),
                     "movement plan should include path planning");

        reasoning.update_context("warehouse");
        require_true(reasoning.get_current_context() == "warehouse", "context should be updated");

        require_true(reasoning.validate_action_feasibility("move_robot", "lab"), "known action/context should be feasible");
        require_true(!reasoning.validate_action_feasibility("unknown_action", "unknown_context"), "unknown action/context should be infeasible");

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "test-embodied-reasoning failed: " << e.what() << std::endl;
        return 1;
    }
}
