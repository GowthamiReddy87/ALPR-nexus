#!/bin/bash
# ═══════════════════════════════════════
#  ALPR NEXUS — Start Script
# ═══════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     ALPR NEXUS — Startup Sequence      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}[✗] Python3 not found. Install Python 3.10+${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] Python3 found: $(python3 --version)${NC}"

# Install dependencies
echo -e "${YELLOW}[→] Installing backend dependencies...${NC}"
cd "$(dirname "$0")/backend"
pip install -r requirements.txt -q --break-system-packages 2>/dev/null || pip install -r requirements.txt -q
echo -e "${GREEN}[✓] Dependencies ready${NC}"

# Start backend
echo -e "${YELLOW}[→] Starting Flask backend on :5050...${NC}"
python3 app.py &
BACKEND_PID=$!
sleep 2

# Check backend
if curl -s http://localhost:5050/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}[✓] Backend running (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}[✗] Backend failed to start${NC}"
fi

# Start frontend
echo -e "${YELLOW}[→] Starting frontend server on :3000...${NC}"
cd "../frontend"
python3 -m http.server 3000 &
FRONTEND_PID=$!
sleep 1

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🚀 ALPR NEXUS IS RUNNING              ║${NC}"
echo -e "${GREEN}║                                        ║${NC}"
echo -e "${GREEN}║  Frontend: http://localhost:3000       ║${NC}"
echo -e "${GREEN}║  Backend:  http://localhost:5050       ║${NC}"
echo -e "${GREEN}║  API Docs: http://localhost:5050/api/  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Servers stopped.'" INT TERM

wait
