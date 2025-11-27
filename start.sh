#!/bin/bash
# RoDLA Complete Startup Script
# Starts both frontend and backend services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        RoDLA DOCUMENT LAYOUT ANALYSIS - 90s Edition      ║${NC}"
echo -e "${BLUE}║            Startup Script (Frontend + Backend)           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if required directories exist
if [ ! -d "deployment/backend" ]; then
    echo -e "${RED}ERROR: deployment/backend directory not found${NC}"
    exit 1
fi

if [ ! -d "frontend" ]; then
    echo -e "${RED}ERROR: frontend directory not found${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ System check passed${NC}"
echo ""

# Function to handle Ctrl+C
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down RoDLA...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

# Set trap for Ctrl+C
trap cleanup SIGINT

# Check ports
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Start Backend
echo -e "${BLUE}[1/2] Starting Backend API (port 8000)...${NC}"

if check_port 8000; then
    echo -e "${YELLOW}⚠ Port 8000 is already in use${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

cd "$SCRIPT_DIR/deployment/backend"
python3 backend.py > /tmp/rodla_backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Backend failed to start${NC}"
    echo -e "${RED}Check logs: cat /tmp/rodla_backend.log${NC}"
    exit 1
fi

# Start Frontend
echo -e "${BLUE}[2/2] Starting Frontend Server (port 8080)...${NC}"

if check_port 8080; then
    echo -e "${YELLOW}⚠ Port 8080 is already in use${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        kill $BACKEND_PID
        exit 1
    fi
fi

cd "$SCRIPT_DIR/frontend"
python3 server.py > /tmp/rodla_frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
sleep 1

# Summary
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ RoDLA System is Ready!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Access Points:${NC}"
echo -e "  🌐 Frontend:   ${BLUE}http://localhost:8080${NC}"
echo -e "  🔌 Backend:    ${BLUE}http://localhost:8000${NC}"
echo -e "  📚 API Docs:   ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Services:${NC}"
echo -e "  Backend PID: $BACKEND_PID"
echo -e "  Frontend PID: $FRONTEND_PID"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo -e "  Backend:  ${BLUE}tail -f /tmp/rodla_backend.log${NC}"
echo -e "  Frontend: ${BLUE}tail -f /tmp/rodla_frontend.log${NC}"
echo ""
echo -e "${YELLOW}Usage:${NC}"
echo -e "  1. Open ${BLUE}http://localhost:8080${NC} in your browser"
echo -e "  2. Upload a document image"
echo -e "  3. Select analysis mode (Standard or Perturbation)"
echo -e "  4. Click [ANALYZE DOCUMENT]"
echo -e "  5. Download results"
echo ""
echo -e "${YELLOW}Exit:${NC}"
echo -e "  Press ${BLUE}Ctrl+C${NC} to stop all services"
echo ""

# Keep running
wait

