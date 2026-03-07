#!/bin/bash
# ============================================================
# Setup do Ambiente Virtual — NSL-KDD Quantum Analysis
# ============================================================

set -e

VENV_NAME="venv_quantum"
KERNEL_NAME="NSL-KDD Quantum"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo " NSL-KDD Quantum Environment Setup"
echo "========================================"
echo ""

# 1. Verificar Python
echo "[1/6] Verificando Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python 3 não encontrado. Instale com: sudo apt install python3 python3-venv"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "      ✓ $PYTHON_VERSION"

# 2. Criar ambiente virtual
echo "[2/6] Criando ambiente virtual '$VENV_NAME'..."
if [ -d "$VENV_NAME" ]; then
    echo "      ℹ Ambiente '$VENV_NAME' já existe. Pulando criação."
else
    python3 -m venv "$VENV_NAME"
    echo "      ✓ Ambiente criado"
fi

# 3. Ativar ambiente
echo "[3/6] Ativando ambiente virtual..."
source "$VENV_NAME/bin/activate"
echo "      ✓ Ativado: $(which python)"

# 4. Atualizar pip
echo "[4/6] Atualizando pip..."
pip install --upgrade pip --quiet
echo "      ✓ pip atualizado"

# 5. Instalar dependências clássicas
echo "[5/6] Instalando dependências clássicas..."
pip install -r "$PROJECT_DIR/requirements.txt"
echo "      ✓ Dependências clássicas instaladas"

# 6. Instalar dependências quânticas
echo "[6/6] Instalando dependências quânticas (pode demorar alguns minutos)..."
pip install -r "$PROJECT_DIR/requirements_quantum.txt"
echo "      ✓ Dependências quânticas instaladas"

# Registrar kernel Jupyter
echo ""
echo "[Extra] Registrando kernel Jupyter '$KERNEL_NAME'..."
python -m ipykernel install --user --name=nsl-kdd-quantum --display-name="$KERNEL_NAME"
echo "      ✓ Kernel registrado"

echo ""
echo "========================================"
echo " ✅ Setup concluído com sucesso!"
echo "========================================"
echo ""
echo "Para usar o ambiente:"
echo "  source $VENV_NAME/bin/activate"
echo "  jupyter notebook"
echo ""
echo "Para baixar o dataset NSL-KDD:"
echo "  chmod +x data/download_data.sh"
echo "  ./data/download_data.sh"
echo ""
