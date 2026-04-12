#!/bin/bash
# =============================================================================
# setup_hedge_fund.sh
# Sets up the ai-hedge-fund source and patches the data layer to use
# yfinance (free) instead of financialdatasets.ai (paid API).
# Run once before starting hedge_fund_web.py.
# =============================================================================

set -e
echo ""
echo "========================================="
echo "  AI Hedge Fund — Setup"
echo "========================================="

# 1. Clone the repo if not already present
if [ ! -d "hedge_fund_src" ]; then
    echo ""
    echo "[1/4] Cloning virattt/ai-hedge-fund..."
    git clone --depth=1 https://github.com/virattt/ai-hedge-fund.git hedge_fund_src
else
    echo "[1/4] hedge_fund_src/ already exists — skipping clone."
    echo "      (To update, run: cd hedge_fund_src && git pull)"
fi

# 2. Patch: replace financialdatasets.ai API with our yfinance adapter
echo ""
echo "[2/4] Patching data layer (financialdatasets.ai → yfinance)..."
cp yfinance_adapter.py hedge_fund_src/src/tools/api.py
echo "      Done."

# 3. Install Python dependencies
echo ""
echo "[3/4] Installing dependencies..."
pip install -q \
    langchain \
    langgraph \
    langchain-anthropic \
    langchain-openai \
    langchain-core \
    yfinance \
    pandas \
    numpy \
    python-dotenv \
    flask \
    tabulate \
    colorama \
    questionary \
    rich \
    python-dateutil

echo "      Done."

# 4. Verify
echo ""
echo "[4/4] Verifying setup..."
python3 -c "
import sys
sys.path.insert(0, 'hedge_fund_src')
from src.tools.api import get_prices, get_financial_metrics
print('      Data layer: OK')
from src.graph.state import AgentState
print('      Graph state: OK')
from src.main import run_hedge_fund
print('      Main entry:  OK')
"

echo ""
echo "========================================="
echo "  Setup complete!"
echo ""
echo "  Start the web app:"
echo "    export ANTHROPIC_API_KEY='sk-ant-...'"
echo "    python hedge_fund_web.py"
echo "    open http://localhost:5001"
echo "========================================="
echo ""
