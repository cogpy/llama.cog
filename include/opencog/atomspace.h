#ifndef OPENCOG_ATOMSPACE_H
#define OPENCOG_ATOMSPACE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstdint>

namespace opencog {

// Forward declarations
class Atom;
class Node;
class Link;

// Atom types enumeration
enum class atom_type {
    CONCEPT_NODE = 1,
    PREDICATE_NODE = 2,
    VARIABLE_NODE = 3,
    EVALUATION_LINK = 100,
    INHERITANCE_LINK = 101,
    SIMILARITY_LINK = 102,
    IMPLICATION_LINK = 103,
    CONTEXT_LINK = 104,
    EXECUTION_LINK = 105
};

// Truth value structure for probabilistic reasoning
struct truth_value {
    double strength;      // Probability or confidence [0.0, 1.0]
    double confidence;    // Confidence in the strength [0.0, 1.0]

    truth_value(double s = 0.5, double c = 0.0) : strength(s), confidence(c) {}

    bool operator==(const truth_value& other) const {
        return strength == other.strength && confidence == other.confidence;
    }
};

// Attention value for importance tracking
struct attention_value {
    double importance;    // Short-term importance [0.0, 1.0]
    double urgency;      // How urgent processing is [0.0, 1.0]

    attention_value(double i = 0.1, double u = 0.0) : importance(i), urgency(u) {}
};

// Base Atom class - fundamental unit of knowledge
class Atom {
public:
    virtual ~Atom() = default;

    uint64_t get_handle() const { return handle_; }
    atom_type get_type() const { return type_; }
    const truth_value& get_truth_value() const { return truth_value_; }
    const attention_value& get_attention_value() const { return attention_value_; }

    void set_truth_value(const truth_value& tv) { truth_value_ = tv; }
    void set_attention_value(const attention_value& av) { attention_value_ = av; }

    virtual std::string to_string() const = 0;
    virtual bool is_node() const = 0;
    virtual bool is_link() const = 0;

protected:
    Atom(atom_type type, uint64_t handle)
        : type_(type), handle_(handle) {}

    atom_type type_;
    uint64_t handle_;
    truth_value truth_value_;
    attention_value attention_value_;
};

// Node class - atomic concepts/predicates
class Node : public Atom {
public:
    Node(atom_type type, const std::string& name, uint64_t handle)
        : Atom(type, handle), name_(name) {}

    const std::string& get_name() const { return name_; }

    std::string to_string() const override {
        return "(" + std::to_string(static_cast<int>(type_)) + " \"" + name_ + "\")";
    }

    bool is_node() const override { return true; }
    bool is_link() const override { return false; }

private:
    std::string name_;
};

// Link class - relationships between atoms
class Link : public Atom {
public:
    Link(atom_type type, const std::vector<std::shared_ptr<Atom>>& outgoing, uint64_t handle)
        : Atom(type, handle), outgoing_(outgoing) {}

    const std::vector<std::shared_ptr<Atom>>& get_outgoing() const { return outgoing_; }
    size_t get_arity() const { return outgoing_.size(); }

    std::string to_string() const override;

    bool is_node() const override { return false; }
    bool is_link() const override { return true; }

private:
    std::vector<std::shared_ptr<Atom>> outgoing_;
};

// atom_space - knowledge base and reasoning engine
class atom_space {
public:
    atom_space();
    ~atom_space() = default;

    // Atom creation and retrieval
    std::shared_ptr<Node> add_node(atom_type type, const std::string& name);
    std::shared_ptr<Link> add_link(atom_type type, const std::vector<std::shared_ptr<Atom>>& outgoing);

    std::shared_ptr<Atom> get_atom(uint64_t handle) const;
    std::vector<std::shared_ptr<Atom>> get_atoms_by_type(atom_type type) const;
    std::vector<std::shared_ptr<Atom>> get_atoms_by_name(const std::string& name) const;

    // Query and pattern matching
    std::vector<std::shared_ptr<Atom>> query(const std::string& pattern) const;

    // Attention and memory management
    std::vector<std::shared_ptr<Atom>> get_attentional_focus(size_t max_atoms = 100) const;
    void update_attention_values();
    void decay_attention();

    // Statistics
    size_t size() const;
    size_t get_num_nodes() const;
    size_t get_num_links() const;

    // Serialization
    std::string to_string() const;
    void clear();

private:
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<Atom>> atoms_;
    std::unordered_map<std::string, std::vector<uint64_t>> name_index_;
    std::unordered_map<atom_type, std::vector<uint64_t>> type_index_;
    uint64_t next_handle_;

    uint64_t generate_handle();
    void add_to_indices(std::shared_ptr<Atom> atom);
    void remove_from_indices(std::shared_ptr<Atom> atom);
};

} // namespace opencog

#endif // OPENCOG_ATOMSPACE_H
