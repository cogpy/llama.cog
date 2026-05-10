#include "opencog/atomspace.h"

#include <cmath>
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
        opencog::atom_space atomspace;

        auto cat = atomspace.add_node(opencog::atom_type::CONCEPT_NODE, "cat");
        auto cat_duplicate = atomspace.add_node(opencog::atom_type::CONCEPT_NODE, "cat");
        require_true(cat->get_handle() == cat_duplicate->get_handle(), "duplicate node should reuse handle");

        auto animal = atomspace.add_node(opencog::atom_type::CONCEPT_NODE, "animal");
        std::vector<std::shared_ptr<opencog::atom>> args = {cat, animal};
        auto inheritance = atomspace.add_link(opencog::atom_type::INHERITANCE_LINK, args);
        auto inheritance_duplicate = atomspace.add_link(opencog::atom_type::INHERITANCE_LINK, args);
        require_true(inheritance->get_handle() == inheritance_duplicate->get_handle(), "duplicate link should reuse handle");

        require_true(atomspace.size() == 3, "atom count should include two nodes and one link");
        require_true(atomspace.get_num_nodes() == 2, "node count should be two");
        require_true(atomspace.get_num_links() == 1, "link count should be one");

        cat->set_truth_value(opencog::truth_value(0.9, 0.8));
        cat->set_attention_value(opencog::attention_value(0.2, 0.1));
        atomspace.update_attention_values();
        constexpr double expected_importance = 0.2 + (0.8 * 0.1);
        require_true(std::abs(cat->get_attention_value().importance - expected_importance) < 1e-9,
                     "importance should be updated using truth value confidence");

        auto queried = atomspace.query("cat");
        require_true(!queried.empty(), "query for existing concept should return results");

        atomspace.clear();
        require_true(atomspace.size() == 0, "clear should remove all atoms");

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "test-atomspace failed: " << e.what() << std::endl;
        return 1;
    }
}
