#!/bin/bash
# ============================================================
# Download NSL-KDD Dataset
# ============================================================

set -e

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://raw.githubusercontent.com/defcom17/NSL_KDD/master"

echo "========================================"
echo " Download NSL-KDD Dataset"
echo "========================================"

download_file() {
    local filename="$1"
    local url="$BASE_URL/$filename"
    local dest="$DATA_DIR/$filename"

    if [ -f "$dest" ]; then
        echo "  ℹ $filename já existe. Pulando."
    else
        echo "  ⬇ Baixando $filename..."
        if command -v wget &> /dev/null; then
            wget -q "$url" -O "$dest"
        elif command -v curl &> /dev/null; then
            curl -sL "$url" -o "$dest"
        else
            echo "ERRO: wget ou curl não encontrado."
            exit 1
        fi
        echo "  ✓ $filename baixado"
    fi
}

download_file "KDDTrain+.txt"
download_file "KDDTest+.txt"
download_file "KDDTrain+_20Percent.txt"
download_file "KDDTest-21.txt"

# Criar arquivo de colunas (feature names)
echo "  ✓ Criando arquivo de features..."
cat > "$DATA_DIR/feature_names.txt" << 'EOF'
duration
protocol_type
service
flag
src_bytes
dst_bytes
land
wrong_fragment
urgent
hot
num_failed_logins
logged_in
num_compromised
root_shell
su_attempted
num_root
num_file_creations
num_shells
num_access_files
num_outbound_cmds
is_host_login
is_guest_login
count
srv_count
serror_rate
srv_serror_rate
rerror_rate
srv_rerror_rate
same_srv_rate
diff_srv_rate
srv_diff_host_rate
dst_host_count
dst_host_srv_count
dst_host_same_srv_rate
dst_host_diff_srv_rate
dst_host_same_src_port_rate
dst_host_srv_diff_host_rate
dst_host_serror_rate
dst_host_srv_serror_rate
dst_host_rerror_rate
dst_host_srv_rerror_rate
label
difficulty_level
EOF

echo ""
echo "========================================"
echo " ✅ Dataset NSL-KDD baixado com sucesso!"
echo "========================================"
echo ""
echo "Arquivos disponíveis em: $DATA_DIR"
ls -lh "$DATA_DIR"/*.txt 2>/dev/null || true
echo ""
