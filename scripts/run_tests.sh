#!/bin/bash
# =============================================================================
# Trading System Test Runner
# =============================================================================
# Comprehensive test runner with coverage and reporting options.
#
# Usage:
#   ./scripts/run_tests.sh              # Run all tests
#   ./scripts/run_tests.sh unit         # Run unit tests only
#   ./scripts/run_tests.sh integration  # Run integration tests only
#   ./scripts/run_tests.sh coverage     # Run with coverage report
#   ./scripts/run_tests.sh quick        # Run quick smoke tests
#   ./scripts/run_tests.sh critical     # Run critical path tests only
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Print banner
echo -e "${BLUE}"
echo "=============================================="
echo "   Trading System Test Suite"
echo "=============================================="
echo -e "${NC}"

# Function to run tests
run_tests() {
    local test_type=$1
    local extra_args="${@:2}"

    case $test_type in
        "unit")
            echo -e "${YELLOW}Running unit tests...${NC}"
            pytest tests/unit/ -m unit -v --tb=short $extra_args
            ;;
        "integration")
            echo -e "${YELLOW}Running integration tests...${NC}"
            pytest tests/integration/ -m integration -v --tb=short $extra_args
            ;;
        "coverage")
            echo -e "${YELLOW}Running tests with coverage...${NC}"
            pytest tests/ \
                --cov=execution \
                --cov=strategies \
                --cov=data \
                --cov=research \
                --cov-config=.coveragerc \
                --cov-report=html \
                --cov-report=term-missing \
                --cov-report=xml \
                -v \
                $extra_args
            echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
            ;;
        "quick")
            echo -e "${YELLOW}Running quick smoke tests...${NC}"
            pytest tests/unit/ -m "unit and not slow" -x -v --tb=line $extra_args
            ;;
        "critical")
            echo -e "${YELLOW}Running critical path tests...${NC}"
            pytest tests/ -m critical -v --tb=short $extra_args
            ;;
        "all")
            echo -e "${YELLOW}Running all tests...${NC}"
            pytest tests/ -v --tb=short $extra_args
            ;;
        *)
            echo -e "${YELLOW}Running all tests...${NC}"
            pytest tests/ -v --tb=short $extra_args
            ;;
    esac
}

# Parse command line argument
TEST_TYPE="${1:-all}"
shift 2>/dev/null || true

# Check for pytest
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest not found. Install with: pip install pytest${NC}"
    exit 1
fi

# Run tests
echo -e "${BLUE}Test type: ${TEST_TYPE}${NC}"
echo ""

if run_tests "$TEST_TYPE" "$@"; then
    echo ""
    echo -e "${GREEN}=============================================="
    echo "   All tests passed!"
    echo -e "==============================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}=============================================="
    echo "   Some tests failed!"
    echo -e "==============================================${NC}"
    exit 1
fi
