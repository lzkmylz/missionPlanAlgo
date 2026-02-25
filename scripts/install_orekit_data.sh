#!/bin/bash
#
# Orekit数据文件安装脚本
#
# 下载和安装Orekit Java后端所需的jar包和数据文件
#
# 使用方法:
#   ./install_orekit_data.sh [安装目录]
#
# 默认安装目录: /usr/local/share/orekit
#

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认安装目录
INSTALL_DIR="${1:-/usr/local/share/orekit}"
DRY_RUN=false

# Orekit版本
OREKIT_VERSION="12.0"
HIPPARCHUS_VERSION="3.0"

# 下载URL
MAVEN_CENTRAL="https://repo1.maven.org/maven2"
OREKIT_URL="${MAVEN_CENTRAL}/org/orekit/orekit/${OREKIT_VERSION}/orekit-${OREKIT_VERSION}.jar"

# Hipparchus jars
HIPPARCHUS_JARS=(
    "hipparchus-core"
    "hipparchus-geometry"
    "hipparchus-ode"
    "hipparchus-filtering"
)

# 数据文件URL
IERS_URL="https://datacenter.iers.org/data/latestVersion/9_FINALS.ALL_IAU2000_V2013_0110.txt"
EGM96_URL="https://earth-info.nga.mil/php/download.php?file=egm-96"
DE440_URL="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"

# 打印帮助信息
print_usage() {
    echo "用法: $0 [选项] [安装目录]"
    echo ""
    echo "选项:"
    echo "  -h, --help      显示帮助信息"
    echo "  -d, --dry-run   试运行，不实际下载文件"
    echo ""
    echo "默认安装目录: /usr/local/share/orekit"
    echo ""
    echo "环境变量:"
    echo "  OREKIT_DATA_DIR  覆盖默认安装目录"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            print_usage
            exit 1
            ;;
        *)
            INSTALL_DIR="$1"
            shift
            ;;
    esac
done

# 检查环境变量
if [ -n "$OREKIT_DATA_DIR" ]; then
    INSTALL_DIR="$OREKIT_DATA_DIR"
fi

echo "=========================================="
echo "Orekit数据文件安装脚本"
echo "=========================================="
echo "安装目录: $INSTALL_DIR"
echo "Orekit版本: $OREKIT_VERSION"
echo "Hipparchus版本: $HIPPARCHUS_VERSION"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}模式: 试运行 (dry-run)${NC}"
fi
echo "=========================================="

# 检查依赖命令
check_dependencies() {
    local missing=()

    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        missing+=("curl 或 wget")
    fi

    if ! command -v java &> /dev/null; then
        missing+=("java")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}错误: 缺少以下依赖:${NC}"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi

    # 检查Java版本
    JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
    echo "检测到Java版本: $JAVA_VERSION"
}

# 创建目录结构
create_directories() {
    echo ""
    echo "创建目录结构..."

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 将创建以下目录:"
        echo "  - $INSTALL_DIR"
        echo "  - $INSTALL_DIR/IERS"
        echo "  - $INSTALL_DIR/EGM96"
        echo "  - $INSTALL_DIR/DE440"
        return
    fi

    mkdir -p "$INSTALL_DIR"/{IERS,EGM96,DE440}
    echo -e "${GREEN}目录创建完成${NC}"
}

# 下载文件函数
download_file() {
    local url=$1
    local output=$2
    local description=$3

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 将下载: $description"
        echo "  URL: $url"
        echo "  输出: $output"
        return 0
    fi

    echo ""
    echo "下载: $description"

    if [ -f "$output" ]; then
        echo "  文件已存在，跳过下载"
        return 0
    fi

    if command -v curl &> /dev/null; then
        if ! curl -L --progress-bar -o "$output" "$url"; then
            echo -e "${RED}错误: 下载失败 $url${NC}"
            return 1
        fi
    elif command -v wget &> /dev/null; then
        if ! wget --progress=bar:force -O "$output" "$url"; then
            echo -e "${RED}错误: 下载失败 $url${NC}"
            return 1
        fi
    else
        echo -e "${RED}错误: 需要curl或wget${NC}"
        return 1
    fi

    echo -e "${GREEN}  下载完成${NC}"
    return 0
}

# 下载Orekit jar
download_orekit() {
    echo ""
    echo "下载Orekit库..."
    local output="$INSTALL_DIR/orekit-${OREKIT_VERSION}.jar"
    download_file "$OREKIT_URL" "$output" "orekit-${OREKIT_VERSION}.jar"
}

# 下载Hipparchus jars
download_hipparchus() {
    echo ""
    echo "下载Hipparchus库..."

    for jar in "${HIPPARCHUS_JARS[@]}"; do
        local url="${MAVEN_CENTRAL}/org/hipparchus/${jar}/${HIPPARCHUS_VERSION}/${jar}-${HIPPARCHUS_VERSION}.jar"
        local output="$INSTALL_DIR/${jar}-${HIPPARCHUS_VERSION}.jar"
        download_file "$url" "$output" "${jar}-${HIPPARCHUS_VERSION}.jar"
    done
}

# 下载数据文件
download_data_files() {
    echo ""
    echo "下载数据文件..."

    # IERS finals.all
    local finals_url="https://datacenter.iers.org/products/eop/rapid/daily/finals2000A.all"
    local finals_output="$INSTALL_DIR/IERS/finals.all"

    if [ "$DRY_RUN" = false ]; then
        echo "下载IERS finals.all (地球自转数据)..."
        if command -v curl &> /dev/null; then
            curl -L -o "$finals_output" "$finals_url" 2>/dev/null || \
                echo -e "${YELLOW}警告: 无法下载finals.all，请手动下载${NC}"
        fi
    else
        echo "[DRY-RUN] 将下载IERS finals.all"
    fi

    # 注意：EGM96和DE440文件较大，可能需要手动下载
    echo ""
    echo -e "${YELLOW}注意: EGM96和DE440文件较大，建议手动下载:${NC}"
    echo "  EGM96: $EGM96_URL"
    echo "    -> $INSTALL_DIR/EGM96/egm96.gfc"
    echo "  DE440: $DE440_URL"
    echo "    -> $INSTALL_DIR/DE440/de440.bsp"
}

# 编译辅助jar
compile_helper_jar() {
    echo ""
    echo "编译Orekit辅助类..."

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local compile_script="$script_dir/compile_helper_jar.sh"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 将执行: $compile_script"
        return
    fi

    if [ -f "$compile_script" ]; then
        bash "$compile_script" "$INSTALL_DIR"
    else
        echo -e "${YELLOW}警告: 编译脚本不存在: $compile_script${NC}"
    fi
}

# 设置权限
set_permissions() {
    if [ "$DRY_RUN" = true ]; then
        return
    fi

    echo ""
    echo "设置文件权限..."

    # 设置目录权限
    chmod -R 755 "$INSTALL_DIR"

    echo -e "${GREEN}权限设置完成${NC}"
}

# 打印安装摘要
print_summary() {
    echo ""
    echo "=========================================="
    echo "安装摘要"
    echo "=========================================="

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}试运行完成，未实际安装文件${NC}"
        return
    fi

    echo "安装目录: $INSTALL_DIR"
    echo ""
    echo "已安装文件:"
    ls -lh "$INSTALL_DIR"/*.jar 2>/dev/null || echo "  (无jar文件)"
    echo ""
    echo "数据目录:"
    echo "  IERS: $INSTALL_DIR/IERS"
    echo "  EGM96: $INSTALL_DIR/EGM96"
    echo "  DE440: $INSTALL_DIR/DE440"
    echo ""
    echo -e "${GREEN}安装完成!${NC}"
    echo ""
    echo "环境变量设置建议:"
    echo "  export OREKIT_DATA_DIR=$INSTALL_DIR"
    echo ""
    echo "验证安装:"
    echo "  python -c \"from core.orbit.visibility.orekit_config import get_orekit_data_dir; print(get_orekit_data_dir())\""
}

# 主函数
main() {
    check_dependencies
    create_directories
    download_orekit
    download_hipparchus
    download_data_files
    compile_helper_jar
    set_permissions
    print_summary
}

# 执行主函数
main
