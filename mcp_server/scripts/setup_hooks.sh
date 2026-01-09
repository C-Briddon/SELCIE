#!/bin/bash
# Setup git hooks for MCP server development
#
# Run this script from anywhere in the SELCIE repo to install
# the pre-commit hook that auto-generates tool documentation.

set -e

# Find repo root
REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"
MCP_HOOK="$REPO_ROOT/mcp_server/.githooks/pre-commit"

echo "Setting up MCP server git hooks..."

# Create symlink
if [ -f "$HOOKS_DIR/pre-commit" ]; then
    echo "Existing pre-commit hook found. Backing up to pre-commit.bak"
    mv "$HOOKS_DIR/pre-commit" "$HOOKS_DIR/pre-commit.bak"
fi

ln -sf "../../mcp_server/.githooks/pre-commit" "$HOOKS_DIR/pre-commit"

echo "Done! Pre-commit hook installed."
echo "When you commit changes to mcp_server/tools/*.py, TOOLS.md will be auto-updated."
