#!/bin/bash

# PR Verification Script
# Ensures the inference branch is ready for PR without weight files

echo "🔍 RoDLA Inference Branch - PR Readiness Check"
echo "================================================"
echo ""

# Check 1: Verify on inference branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "inference" ]; then
    echo "❌ ERROR: Not on inference branch (current: $CURRENT_BRANCH)"
    exit 1
fi
echo "✅ On inference branch"
echo ""

# Check 2: Verify no uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ ERROR: Uncommitted changes detected:"
    git status --short
    exit 1
fi
echo "✅ No uncommitted changes"
echo ""

# Check 3: Check for weight files (should be empty or deleted)
echo "📋 Checking for weight files..."
WEIGHT_FILES=$(git diff --name-only main...inference | grep -E '\.(pth|pt|ckpt|weights|pkl)$' || echo "")

if [ -z "$WEIGHT_FILES" ]; then
    echo "✅ No weight files added in diff"
else
    echo "⚠️  Found weight files in diff:"
    echo "$WEIGHT_FILES"
    
    # Check if they're empty
    for file in $WEIGHT_FILES; do
        SIZE=$(git diff main...inference -- "$file" | wc -c)
        if [ "$SIZE" -gt 1000 ]; then
            echo "❌ ERROR: $file has size changes (likely a real weight file)"
            exit 1
        else
            echo "   ℹ️  $file is empty/minimal (safe)"
        fi
    done
fi
echo ""

# Check 4: Summary of changes
echo "📊 Change Summary:"
git diff --stat main...inference | tail -1
echo ""

# Check 5: Commit count
COMMITS=$(git rev-list --count main..inference)
echo "📝 Commits on inference: $COMMITS"
echo ""

# Check 6: Key files modified
echo "🔧 Key Files Modified:"
git diff --name-only main...inference | grep -E '(backend\.py|script\.js|index\.html|perturbations)' | sed 's/^/   ✓ /'
echo ""

# Success message
echo "================================================"
echo "✅ PR READINESS CHECK PASSED"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Push to GitHub: git push origin inference"
echo "2. Go to GitHub and create a Pull Request"
echo "3. Title: 'feat: add complete inference pipeline with web UI'"
echo "4. Description: See PR_SUMMARY.md"
echo ""
echo "Commands:"
echo "  - View PR changes: git diff main...inference"
echo "  - View commits: git log main...inference"
echo "  - Size estimate: git diff --stat main...inference | tail -1"
