#include "opencog/atomspace.h"
#include <algorithm>
#include <sstream>
#include <random>

namespace opencog {

// Link implementation
std::string Link::to_string() const {
    std::stringstream ss;
    ss << "(" << static_cast<int>(type_);
    for (const auto& atom : outgoing_) {
        ss << " " << atom->to_string();
    }
    ss << ")";
    return ss.str();
}

// AtomSpace implementation
AtomSpace::AtomSpace() : next_handle_(1) {}

std::shared_ptr<Node> AtomSpace::add_node(AtomType type, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if node already exists
    auto it = name_index_.find(name);
    if (it != name_index_.end()) {
        for (uint64_t handle : it->second) {
            auto atom = atoms_[handle];
            if (atom->get_type() == type && atom->is_node()) {
                return std::static_pointer_cast<Node>(atom);
            }
        }
    }

    // Create new node
    uint64_t handle = generate_handle();
    auto node = std::make_shared<Node>(type, name, handle);
    atoms_[handle] = node;
    add_to_indices(node);

    return node;
}

std::shared_ptr<Link> AtomSpace::add_link(AtomType type, const std::vector<std::shared_ptr<Atom>>& outgoing) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if link already exists (based on type and outgoing set)
    for (const auto& [handle, atom] : atoms_) {
        if (atom->get_type() == type && atom->is_link()) {
            auto link = std::static_pointer_cast<Link>(atom);
            if (link->get_outgoing() == outgoing) {
                return link;
            }
        }
    }

    // Create new link
    uint64_t handle = generate_handle();
    auto link = std::make_shared<Link>(type, outgoing, handle);
    atoms_[handle] = link;
    add_to_indices(link);

    return link;
}

std::shared_ptr<Atom> AtomSpace::get_atom(uint64_t handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = atoms_.find(handle);
    return (it != atoms_.end()) ? it->second : nullptr;
}

std::vector<std::shared_ptr<Atom>> AtomSpace::get_atoms_by_type(AtomType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Atom>> result;

    auto it = type_index_.find(type);
    if (it != type_index_.end()) {
        for (uint64_t handle : it->second) {
            auto atom = atoms_.at(handle);
            result.push_back(atom);
        }
    }

    return result;
}

std::vector<std::shared_ptr<Atom>> AtomSpace::get_atoms_by_name(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Atom>> result;

    auto it = name_index_.find(name);
    if (it != name_index_.end()) {
        for (uint64_t handle : it->second) {
            auto atom = atoms_.at(handle);
            result.push_back(atom);
        }
    }

    return result;
}

std::vector<std::shared_ptr<Atom>> AtomSpace::query(const std::string& pattern) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Atom>> result;

    // Simple pattern matching - look for atoms containing the pattern
    for (const auto& [handle, atom] : atoms_) {
        std::string atom_str = atom->to_string();
        if (atom_str.find(pattern) != std::string::npos) {
            result.push_back(atom);
        }
    }

    return result;
}

std::vector<std::shared_ptr<Atom>> AtomSpace::get_attentional_focus(size_t max_atoms) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Atom>> all_atoms;

    for (const auto& [handle, atom] : atoms_) {
        all_atoms.push_back(atom);
    }

    // Sort by attention value (importance + urgency).
    // A sum is used rather than a product so that atoms with default urgency
    // (0.0) can still be ranked by their importance.
    std::sort(all_atoms.begin(), all_atoms.end(),
              [](const std::shared_ptr<Atom>& a, const std::shared_ptr<Atom>& b) {
                  const auto& av_a = a->get_attention_value();
                  const auto& av_b = b->get_attention_value();
                  return (av_a.importance + av_a.urgency) > (av_b.importance + av_b.urgency);
              });

    if (all_atoms.size() > max_atoms) {
        all_atoms.resize(max_atoms);
    }

    return all_atoms;
}

void AtomSpace::update_attention_values() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Simple attention updating - boost importance of recently accessed atoms
    for (auto& [handle, atom] : atoms_) {
        AttentionValue av = atom->get_attention_value();

        // Increase importance based on truth value confidence
        double tv_boost = atom->get_truth_value().confidence * 0.1;
        av.importance = std::min(1.0, av.importance + tv_boost);

        atom->set_attention_value(av);
    }
}

void AtomSpace::decay_attention() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [handle, atom] : atoms_) {
        AttentionValue av = atom->get_attention_value();

        // Decay attention over time
        av.importance *= 0.99;  // 1% decay per call
        av.urgency *= 0.95;     // 5% decay per call

        atom->set_attention_value(av);
    }
}

size_t AtomSpace::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return atoms_.size();
}

size_t AtomSpace::get_num_nodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& [handle, atom] : atoms_) {
        if (atom->is_node()) count++;
    }
    return count;
}

size_t AtomSpace::get_num_links() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& [handle, atom] : atoms_) {
        if (atom->is_link()) count++;
    }
    return count;
}

std::string AtomSpace::to_string() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;

    ss << "AtomSpace[" << atoms_.size() << " atoms]\n";
    for (const auto& [handle, atom] : atoms_) {
        ss << "  " << atom->to_string() << " (TV: "
           << atom->get_truth_value().strength << ","
           << atom->get_truth_value().confidence << ")\n";
    }

    return ss.str();
}

void AtomSpace::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    atoms_.clear();
    name_index_.clear();
    type_index_.clear();
    next_handle_ = 1;
}

uint64_t AtomSpace::generate_handle() {
    return next_handle_++;
}

void AtomSpace::add_to_indices(std::shared_ptr<Atom> atom) {
    uint64_t handle = atom->get_handle();

    // Type index
    type_index_[atom->get_type()].push_back(handle);

    // Name index (for nodes)
    if (atom->is_node()) {
        auto node = std::static_pointer_cast<Node>(atom);
        name_index_[node->get_name()].push_back(handle);
    }
}

void AtomSpace::remove_from_indices(std::shared_ptr<Atom> atom) {
    uint64_t handle = atom->get_handle();

    // Remove from type index
    auto& type_vec = type_index_[atom->get_type()];
    type_vec.erase(std::remove(type_vec.begin(), type_vec.end(), handle), type_vec.end());

    // Remove from name index (for nodes)
    if (atom->is_node()) {
        auto node = std::static_pointer_cast<Node>(atom);
        auto& name_vec = name_index_[node->get_name()];
        name_vec.erase(std::remove(name_vec.begin(), name_vec.end(), handle), name_vec.end());
    }
}

} // namespace opencog
