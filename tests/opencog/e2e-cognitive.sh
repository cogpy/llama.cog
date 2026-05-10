#!/usr/bin/env bash
# End-to-end tests for the llama-cognitive binary.
#
# Usage:
#   e2e-cognitive.sh <path-to-llama-cognitive> <path-to-model.gguf>
#
# The script feeds scripted commands to the binary via stdin and checks that
# expected strings appear in the output.  It exits with a non-zero status if
# any check fails.

set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <llama-cognitive> <model.gguf>"
    exit 1
fi

BINARY=$1
MODEL=$2

if [ ! -x "$BINARY" ]; then
    echo "ERROR: binary not found or not executable: $BINARY"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model file not found: $MODEL"
    exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; exit 1; }

# Run the binary with stdin piped from a here-doc, capture all output.
run_cognitive() {
    timeout 60 "$BINARY" -m "$MODEL" "$@"
}

# ---------------------------------------------------------------------------
# 1. --help / startup smoke-test
# ---------------------------------------------------------------------------
echo "--- Test 1: startup and help command ---"
OUTPUT=$(printf 'help\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Commands:" || fail "help output missing 'Commands:'"
echo "$OUTPUT" | grep -q "status" || fail "help output missing 'status' command"
echo "$OUTPUT" | grep -q "goals" || fail "help output missing 'goals' command"
pass "startup and help"

# ---------------------------------------------------------------------------
# 2. status command
# ---------------------------------------------------------------------------
echo "--- Test 2: status command ---"
OUTPUT=$(printf 'status\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "System Status" || fail "status output missing 'System Status'"
echo "$OUTPUT" | grep -q "AtomSpace size" || fail "status output missing 'AtomSpace size'"
echo "$OUTPUT" | grep -q "Active goals" || fail "status output missing 'Active goals'"
echo "$OUTPUT" | grep -q "Cognitive cycles" || fail "status output missing 'Cognitive cycles'"
pass "status"

# ---------------------------------------------------------------------------
# 3. goals command
# ---------------------------------------------------------------------------
echo "--- Test 3: goals command ---"
OUTPUT=$(printf 'goals\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Active Goals" || fail "goals output missing 'Active Goals'"
pass "goals"

# ---------------------------------------------------------------------------
# 4. add concept command
# ---------------------------------------------------------------------------
echo "--- Test 4: add concept ---"
OUTPUT=$(printf 'add test_concept\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Added concept: test_concept" || fail "add concept output missing confirmation"
pass "add concept"

# ---------------------------------------------------------------------------
# 5. goal command
# ---------------------------------------------------------------------------
echo "--- Test 5: set goal ---"
OUTPUT=$(printf 'goal explore the environment\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Added goal: explore the environment" || fail "goal output missing confirmation"
pass "set goal"

# ---------------------------------------------------------------------------
# 6. spatial command
# ---------------------------------------------------------------------------
echo "--- Test 6: spatial knowledge ---"
OUTPUT=$(printf 'spatial robot lab\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Added spatial knowledge: robot at lab" || fail "spatial output missing confirmation"
pass "spatial knowledge"

# ---------------------------------------------------------------------------
# 7. causal command
# ---------------------------------------------------------------------------
echo "--- Test 7: causal relationship ---"
OUTPUT=$(printf 'causal learning improvement\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Added causal relation: learning -> improvement" || fail "causal output missing confirmation"
pass "causal relationship"

# ---------------------------------------------------------------------------
# 8. knowledge command (after adding concepts)
# ---------------------------------------------------------------------------
echo "--- Test 8: knowledge base display ---"
OUTPUT=$(printf 'add knowledge_atom\nknowledge\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Knowledge Base" || fail "knowledge output missing 'Knowledge Base'"
echo "$OUTPUT" | grep -q "Total atoms" || fail "knowledge output missing 'Total atoms'"
pass "knowledge base display"

# ---------------------------------------------------------------------------
# 9. cycle command
# ---------------------------------------------------------------------------
echo "--- Test 9: run single cycle ---"
OUTPUT=$(printf 'cycle\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Running cognitive cycle" || fail "cycle output missing 'Running cognitive cycle'"
echo "$OUTPUT" | grep -q "Cycle completed" || fail "cycle output missing 'Cycle completed'"
pass "single cycle"

# ---------------------------------------------------------------------------
# 10. plan command
# ---------------------------------------------------------------------------
echo "--- Test 10: plan actions ---"
OUTPUT=$(printf 'plan move robot\nquit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Action plan" || fail "plan output missing 'Action plan'"
pass "plan actions"

# ---------------------------------------------------------------------------
# 11. save / load knowledge
# ---------------------------------------------------------------------------
echo "--- Test 11: save and load knowledge ---"
KNOWLEDGE_FILE="$TMP_DIR/knowledge.json"
OUTPUT=$(printf "add save_test_concept\nsave %s\nload %s\nknowledge\nquit\n" "$KNOWLEDGE_FILE" "$KNOWLEDGE_FILE" | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Knowledge saved" || fail "save output missing confirmation"
[ -f "$KNOWLEDGE_FILE" ] || fail "knowledge file was not created on disk"
echo "$OUTPUT" | grep -q "Knowledge loaded" || fail "load output missing confirmation"
echo "$OUTPUT" | grep -q "Total atoms" || fail "knowledge display after load missing 'Total atoms'"
pass "save and load knowledge"

# ---------------------------------------------------------------------------
# 12. exit via 'exit' command
# ---------------------------------------------------------------------------
echo "--- Test 12: exit command ---"
OUTPUT=$(printf 'exit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Shutting down" || fail "exit output missing 'Shutting down'"
pass "exit command"

# ---------------------------------------------------------------------------
# 13. quit alias
# ---------------------------------------------------------------------------
echo "--- Test 13: quit alias ---"
OUTPUT=$(printf 'quit\n' | run_cognitive 2>&1)
echo "$OUTPUT" | grep -q "Shutting down" || fail "quit output missing 'Shutting down'"
pass "quit alias"

echo ""
echo "All E2E tests passed."
