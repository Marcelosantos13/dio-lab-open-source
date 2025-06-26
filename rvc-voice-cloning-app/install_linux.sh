#!/bin/bash

echo "========================================"
echo " RVC Voice Cloning App - Instalador"
echo "========================================"
echo

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python3 não encontrado!"
    echo "Por favor, instale Python 3.8+ usando:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

echo "Python3 encontrado!"
echo

# Verificar pip
if ! command -v pip3 &> /dev/null; then
    echo "Instalando pip..."
    python3 -m ensurepip --upgrade
fi

echo "Instalando dependências..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERRO: Falha na instalação das dependências!"
    exit 1
fi

# Tornar script executável
chmod +x executar.sh

echo
echo "========================================"
echo " Instalação concluída com sucesso!"
echo "========================================"
echo
echo "Para executar o aplicativo:"
echo "  1. Execute: python3 main.py"
echo "  2. Ou execute: ./executar.sh"
echo