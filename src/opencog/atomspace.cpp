#include "opencog/atomspace.h"
#include <algorithm>
#include <sstream>
#include <random>

namespace opencog {

// link implementation
std::string link::to_string() const {
    std::stringstream ss;
    ss << "(" << static_cast<int>(type_);
    for (const auto& a : outgoing_) {
        ss << " " << a->to_string();
    }
    ss << ")";
    return ss.str();
}

// atom_space implementation
atom_space::atom_space() : next_handle_(1) {}

std::shared_ptr<node> atom_space::add_node(atom_type type, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if node already exists
    auto it = name_index_.find(name);
    if (it != name_index_.end()) {
        for (uint64_t handle : it->second) {
            auto a = atoms_[handle];
            if (a->get_type() == type && a->is_node()) {
                return std::static_pointer_cast<node>(a);
            }
        }
    }

    // Create new node
    uint64_t handle = generate_handle();
    auto n = std::make_shared<node>(type, name, handle);
    atoms_[handle] = n;
    add_to_indices(n);

    return n;
}

std::shared_ptr<link> atom_space::add_link(atom_type type, const std::vector<std::shared_ptr<atom>>& outgoing) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if link already exists (based on type and outgoing set)
    for (const auto & kv : atoms_) {
        const auto & a = kv.second;
        if (a->get_type() == type && a->is_link()) {
            auto lk = std::static_pointer_cast<link>(a);
            if (lk->get_outgoing() == outgoing) {
                return lk;
            }
        }
    }

    // Create new link
    uint64_t handle = generate_handle();
    auto lk = std::make_shared<link>(type, outgoing, handle);
    atoms_[handle] = lk;
    add_to_indices(lk);

    return lk;
}

std::shared_ptr<atom> atom_space::get_atom(uint64_t handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = atoms_.find(handle);
    return (it != atoms_.end()) ? it->second : nullptr;
}

std::vector<std::shared_ptr<atom>> atom_space::get_atoms_by_type(atom_type type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<atom>> result;

    auto it = type_index_.find(type);
    if (it != type_index_.end()) {
        for (uint64_t handle : it->second) {
            auto a = atoms_.at(handle);
            result.push_back(a);
        }
    }

    return result;
}

std::vector<std::shared_ptr<atom>> atom_space::get_atoms_by_name(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<atom>> result;

    auto it = name_index_.find(name);
    if (it != name_index_.end()) {
        for (uint64_t handle : it->second) {
            auto a = atoms_.at(handle);
            result.push_back(a);
        }
    }

    return result;
}

std::vector<std::shared_ptr<atom>> atom_space::query(const std::string& pattern) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<atom>> result;

    // Simple pattern matching - look for atoms containing the pattern
    for (const auto & kv : atoms_) {
        const auto & a = kv.second;
        std::string atom_str = a->to_string();
        if (atom_str.find(pattern) != std::string::npos) {
            result.push_back(a);
        }
    }

    return result;
}

std::vector<std::shared_ptr<atom>> atom_space::get_attentional_focus(size_t max_atoms) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<atom>> all_atoms;

    for (const auto & kv : atoms_) {
        all_atoms.push_back(kv.second);
    }

    // Sort by attention value (importance + urgency).
    // A sum is used rather than a product so that atoms with default urgency
    // (0.0) can still be ranked by their importance.
    std::sort(all_atoms.begin(), all_atoms.end(),
              [](const std::shared_ptr<atom>& a, const std::shared_ptr<atom>& b) {
                  const auto& av_a = a->get_attention_value();
                  const auto& av_b = b->get_attention_value();
                  return (av_a.importance + av_a.urgency) > (av_b.importance + av_b.urgency);
              });

    if (all_atoms.size() > max_atoms) {
        all_atoms.resize(max_atoms);
    }

    return all_atoms;
}

void atom_space::update_attention_values() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Simple attention updating - boost importance of recently accessed atoms
    for (auto & kv : atoms_) {
        auto & a = kv.second;
        attention_value av = a->get_attention_value();

        // Increase importance based on truth value confidence
        double tv_boost = a->get_truth_value().confidence * 0.1;
        av.importance = std::min(1.0, av.importance + tv_boost);

        a->set_attention_value(av);
    }
}

void atom_space::decay_attention() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto & kv : atoms_) {
        auto & a = kv.second;
        attention_value av = a->get_attention_value();

        // Decay attention over time
        av.importance *= 0.99;  // 1% decay per call
        av.urgency *= 0.95;     // 5% decay per call

        a->set_attention_value(av);
    }
}

size_t atom_space::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return atoms_.size();
}

size_t atom_space::get_num_nodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto & kv : atoms_) {
        if (kv.second->is_node()) count++;
    }
    return count;
}

size_t atom_space::get_num_links() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto & kv : atoms_) {
        if (kv.second->is_link()) count++;
    }
    return count;
}

std::string atom_space::to_string() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;

    ss << "AtomSpace[" << atoms_.size() << " atoms]\n";
    for (const auto & kv : atoms_) {
        const auto & a = kv.second;
        ss << "  " << a->to_string() << " (TV: "
           << a->get_truth_value().strength << ","
           << a->get_truth_value().confidence << ")\n";
    }

    return ss.str();
}

void atom_space::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    atoms_.clear();
    name_index_.clear();
    type_index_.clear();
    next_handle_ = 1;
}

uint64_t atom_space::generate_handle() {
    return next_handle_++;
}

void atom_space::add_to_indices(std::shared_ptr<atom> a) {
    uint64_t handle = a->get_handle();

    // Type index
    type_index_[a->get_type()].push_back(handle);

    // Name index (for nodes)
    if (a->is_node()) {
        auto n = std::static_pointer_cast<node>(a);
        name_index_[n->get_name()].push_back(handle);
    }
}

void atom_space::remove_from_indices(std::shared_ptr<atom> a) {
    uint64_t handle = a->get_handle();

    // Remove from type index
    auto& type_vec = type_index_[a->get_type()];
    type_vec.erase(std::remove(type_vec.begin(), type_vec.end(), handle), type_vec.end());

    // Remove from name index (for nodes)
    if (a->is_node()) {
        auto n = std::static_pointer_cast<node>(a);
        auto& name_vec = name_index_[n->get_name()];
        name_vec.erase(std::remove(name_vec.begin(), name_vec.end(), handle), name_vec.end());
    }
}

} // namespace opencog
