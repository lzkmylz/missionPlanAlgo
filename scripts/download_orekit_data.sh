#!/bin/bash
#
# Orekit 数据文件一键下载脚本
#
# 下载 Orekit Java 后端运行所需的所有必要数据文件，包括：
# - Orekit/Hipparchus JAR 包
# - EGM2008 高精度重力场数据（必须）
# - IERS 地球自转参数
# - DE440 行星历表
#
# 使用方法:
#   ./download_orekit_data.sh [选项]
#
# 默认安装目录: ~/orekit-data
#

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 版本配置
OREKIT_VERSION="12.0"
HIPPARCHUS_VERSION="3.0"

# 默认安装目录
INSTALL_DIR="${HOME}/orekit-data"

# 下载源配置
MAVEN_CENTRAL="https://repo1.maven.org/maven2"
NGA_BASE="https://earth-info.nga.mil/php/download.php"
IERS_BASE="https://datacenter.iers.org/products/eop/rapid/standard"
NAIF_BASE="https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

# 命令行选项
FORCE=false
VERIFY_ONLY=false
DRY_RUN=false

# 打印帮助信息
print_usage() {
    cat << 'EOF'
用法: download_orekit_data.sh [选项]

选项:
  -h, --help          显示帮助信息
  -d, --dir DIR       指定安装目录 (默认: ~/orekit-data)
  -f, --force         强制重新下载，覆盖已有文件
  --verify            仅验证现有数据完整性
  --dry-run           试运行，显示将要执行的操作

环境变量:
  OREKIT_DATA_DIR     覆盖默认安装目录
  HTTP_PROXY          HTTP代理设置
  HTTPS_PROXY         HTTPS代理设置

示例:
  # 默认安装到 ~/orekit-data
  ./download_orekit_data.sh

  # 安装到指定目录
  ./download_orekit_data.sh -d /opt/orekit-data

  # 强制重新下载所有文件
  ./download_orekit_data.sh -f

  # 验证现有数据
  ./download_orekit_data.sh --verify

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -d|--dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            print_usage
            exit 1
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# 应用环境变量
if [ -n "$OREKIT_DATA_DIR" ]; then
    INSTALL_DIR="$OREKIT_DATA_DIR"
fi

# 检查依赖命令
check_dependencies() {
    echo "检查依赖..."

    local missing=()

    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        missing+=("curl 或 wget")
    fi

    if ! command -v unzip &> /dev/null; then
        missing+=("unzip")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}错误: 缺少以下依赖:${NC}"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "请安装依赖后重试:"
        echo "  Ubuntu/Debian: sudo apt-get install curl unzip"
        echo "  CentOS/RHEL:   sudo yum install curl unzip"
        echo "  macOS:         brew install curl unzip"
        exit 1
    fi

    echo -e "${GREEN}✓ 依赖检查通过${NC}"
}

# 检查磁盘空间
check_disk_space() {
    local required_mb=500  # 需要约500MB空间

    # 获取目标目录所在分区的可用空间
    local available_kb=$(df -k "$INSTALL_DIR" 2>/dev/null | tail -1 | awk '{print $4}')
    if [ -z "$available_kb" ]; then
        # 如果目录不存在，检查父目录
        local parent_dir=$(dirname "$INSTALL_DIR")
        available_kb=$(df -k "$parent_dir" 2>/dev/null | tail -1 | awk '{print $4}')
    fi

    local available_mb=$((available_kb / 1024))

    if [ "$available_mb" -lt "$required_mb" ]; then
        echo -e "${RED}错误: 磁盘空间不足${NC}"
        echo "需要: ${required_mb}MB"
        echo "可用: ${available_mb}MB"
        exit 1
    fi
}

# 创建目录结构
create_directories() {
    echo ""
    echo "创建目录结构: $INSTALL_DIR"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 将创建以下目录:"
        echo "  - $INSTALL_DIR"
        echo "  - $INSTALL_DIR/lib"
        echo "  - $INSTALL_DIR/IERS"
        echo "  - $INSTALL_DIR/potential/egm-format"
        echo "  - $INSTALL_DIR/de440"
        return
    fi

    mkdir -p "$INSTALL_DIR"/{lib,IERS,potential/egm-format,de440}
    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# 下载文件（支持断点续传）
download_file() {
    local url=$1
    local output=$2
    local description=$3
    local expected_size=$4  # 可选：预期文件大小（字节）

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 将下载: $description"
        echo "  URL: $url"
        echo "  输出: $output"
        return 0
    fi

    echo ""
    echo "下载: $description"

    # 检查文件是否已存在
    if [ -f "$output" ] && [ "$FORCE" = false ]; then
        local existing_size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
        if [ -n "$expected_size" ] && [ "$existing_size" -eq "$expected_size" ]; then
            echo -e "${GREEN}✓ 文件已存在且大小匹配，跳过下载${NC}"
            return 0
        else
            echo "文件已存在，但大小不匹配或未知，重新下载..."
        fi
    fi

    # 下载文件（支持断点续传）
    local temp_output="${output}.tmp"

    if command -v curl &> /dev/null; then
        if ! curl -C - -L --progress-bar -o "$temp_output" "$url" 2>&1; then
            echo -e "${RED}✗ 下载失败: $url${NC}"
            rm -f "$temp_output"
            return 1
        fi
    elif command -v wget &> /dev/null; then
        if ! wget -c --progress=bar:force -O "$temp_output" "$url" 2>&1; then
            echo -e "${RED}✗ 下载失败: $url${NC}"
            rm -f "$temp_output"
            return 1
        fi
    else
        echo -e "${RED}✗ 需要 curl 或 wget${NC}"
        return 1
    fi

    # 移动临时文件到最终位置
    mv "$temp_output" "$output"
    echo -e "${GREEN}✓ 下载完成${NC}"
    return 0
}

# 下载 Orekit JAR 包
download_orekit_jars() {
    echo ""
    echo "========================================"
    echo "下载 Orekit 库文件"
    echo "========================================"

    local lib_dir="$INSTALL_DIR/lib"

    # Orekit 主库
    local orekit_url="${MAVEN_CENTRAL}/org/orekit/orekit/${OREKIT_VERSION}/orekit-${OREKIT_VERSION}.jar"
    local orekit_output="${lib_dir}/orekit-${OREKIT_VERSION}.jar"

    if [ "$VERIFY_ONLY" = false ]; then
        download_file "$orekit_url" "$orekit_output" "orekit-${OREKIT_VERSION}.jar"
    fi

    # Hipparchus 依赖库
    local hipparchus_libs=("core" "geometry" "ode" "filtering")

    for lib in "${hipparchus_libs[@]}"; do
        local jar_name="hipparchus-${lib}-${HIPPARCHUS_VERSION}.jar"
        local url="${MAVEN_CENTRAL}/org/hipparchus/hipparchus-${lib}/${HIPPARCHUS_VERSION}/${jar_name}"
        local output="${lib_dir}/${jar_name}"

        if [ "$VERIFY_ONLY" = false ]; then
            download_file "$url" "$output" "$jar_name"
        fi
    done

    echo -e "${GREEN}✓ Orekit 库文件下载完成${NC}"
}

# 下载 EGM2008 重力场数据
download_egm2008() {
    echo ""
    echo "========================================"
    echo "下载 EGM2008 重力场数据 (必须)"
    echo "========================================"

    local egm_dir="$INSTALL_DIR/potential/egm-format"
    local egm_file="${egm_dir}/EGM2008_to2190_TideFree.gz"

    if [ "$VERIFY_ONLY" = true ]; then
        if [ -f "$egm_file" ]; then
            local size=$(du -h "$egm_file" | cut -f1)
            echo -e "${GREEN}✓ EGM2008 数据存在 ($size)${NC}"
        else
            echo -e "${RED}✗ EGM2008 数据不存在${NC}"
            return 1
        fi
        return 0
    fi

    # EGM2008 从 NGA 下载
    local egm_url="${NGA_BASE}?file=egm-08spherical"
    local zip_file="${egm_dir}/egm2008_spherical.zip"

    echo -e "${BLUE}数据来源: NGA (National Geospatial-Intelligence Agency)${NC}"
    echo -e "${BLUE}文件大小: 约 104MB (压缩) / 232MB (解压)${NC}"
    echo ""

    # 下载 zip 文件
    if download_file "$egm_url" "$zip_file" "EGM2008 球谐系数 (ZIP)"; then
        echo ""
        echo "解压 EGM2008 数据..."

        # 解压特定文件
        if unzip -q -o "$zip_file" "EGM2008_to2190_TideFree" -d "$egm_dir"; then
            # 压缩为 Orekit 格式
            echo "压缩为 Orekit 格式..."
            gzip -c "${egm_dir}/EGM2008_to2190_TideFree" > "$egm_file"

            # 清理临时文件
            rm -f "$zip_file"
            rm -f "${egm_dir}/EGM2008_to2190_TideFree"

            echo -e "${GREEN}✓ EGM2008 数据准备完成${NC}"
        else
            echo -e "${RED}✗ 解压失败${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ EGM2008 下载失败${NC}"
        echo ""
        echo "可能原因:"
        echo "  - 网络连接问题"
        echo "  - NGA 官网访问受限"
        echo ""
        echo "手动下载步骤:"
        echo "  1. 访问: https://earth-info.nga.mil/"
        echo "  2. 下载: EGM2008 Spherical Harmonics"
        echo "  3. 解压并将 EGM2008_to2190_TideFree 放入:"
        echo "     ${egm_dir}/"
        return 1
    fi
}

# 下载 IERS 地球自转数据
download_iers() {
    echo ""
    echo "========================================"
    echo "下载 IERS 地球自转数据"
    echo "========================================"

    local iers_file="$INSTALL_DIR/IERS/finals2000A.all"
    local iers_url="${IERS_BASE}/finals2000A.all"

    if [ "$VERIFY_ONLY" = true ]; then
        if [ -f "$iers_file" ]; then
            local size=$(du -h "$iers_file" | cut -f1)
            echo -e "${GREEN}✓ IERS 数据存在 ($size)${NC}"
        else
            echo -e "${YELLOW}⚠ IERS 数据不存在${NC}"
        fi
        return 0
    fi

    # IERS 数据经常更新，总是尝试下载最新版本
    if download_file "$iers_url" "$iers_file" "IERS finals2000A.all"; then
        echo -e "${GREEN}✓ IERS 数据下载完成${NC}"
    else
        echo -e "${YELLOW}⚠ IERS 数据下载失败，部分功能可能受影响${NC}"
        echo "  （地球自转参数，用于高精度坐标转换）"
    fi
}

# 下载 DE440 星历表
download_de440() {
    echo ""
    echo "========================================"
    echo "下载 DE440 行星历表"
    echo "========================================"

    local de440_file="$INSTALL_DIR/de440/de440.bsp"
    local de440_url="${NAIF_BASE}/spk/planets/de440.bsp"

    if [ "$VERIFY_ONLY" = true ]; then
        if [ -f "$de440_file" ]; then
            local size=$(du -h "$de440_file" | cut -f1)
            echo -e "${GREEN}✓ DE440 数据存在 ($size)${NC}"
        else
            echo -e "${YELLOW}⚠ DE440 数据不存在${NC}"
        fi
        return 0
    fi

    # DE440 文件很大（约 120MB），如果已存在则跳过
    if [ -f "$de440_file" ] && [ "$FORCE" = false ]; then
        echo -e "${GREEN}✓ DE440 数据已存在，跳过下载${NC}"
        return 0
    fi

    echo -e "${BLUE}数据来源: NASA JPL${NC}"
    echo -e "${BLUE}文件大小: 约 120MB${NC}"
    echo ""

    if download_file "$de440_url" "$de440_file" "DE440 行星历表"; then
        echo -e "${GREEN}✓ DE440 数据下载完成${NC}"
    else
        echo -e "${YELLOW}⚠ DE440 数据下载失败，部分功能可能受影响${NC}"
        echo "  （行星历表，用于日月位置计算）"
    fi
}

# 验证安装
verify_installation() {
    echo ""
    echo "========================================"
    echo "验证安装"
    echo "========================================"

    local errors=0

    # 检查 JAR 包
    echo "检查 JAR 包..."
    local lib_dir="$INSTALL_DIR/lib"
    local required_jars=(
        "orekit-${OREKIT_VERSION}.jar"
        "hipparchus-core-${HIPPARCHUS_VERSION}.jar"
        "hipparchus-geometry-${HIPPARCHUS_VERSION}.jar"
        "hipparchus-ode-${HIPPARCHUS_VERSION}.jar"
    )

    for jar in "${required_jars[@]}"; do
        if [ -f "${lib_dir}/${jar}" ]; then
            echo -e "  ${GREEN}✓${NC} $jar"
        else
            echo -e "  ${RED}✗${NC} $jar (缺失)"
            ((errors++))
        fi
    done

    # 检查 EGM2008（必须）
    echo ""
    echo "检查 EGM2008 数据..."
    local egm_file="$INSTALL_DIR/potential/egm-format/EGM2008_to2190_TideFree.gz"
    if [ -f "$egm_file" ]; then
        local size=$(du -h "$egm_file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} EGM2008 ($size)"
    else
        echo -e "  ${RED}✗${NC} EGM2008 (缺失) [必须]"
        ((errors++))
    fi

    # 检查 IERS
    echo ""
    echo "检查 IERS 数据..."
    if [ -f "$INSTALL_DIR/IERS/finals2000A.all" ]; then
        echo -e "  ${GREEN}✓${NC} finals2000A.all"
    else
        echo -e "  ${YELLOW}⚠${NC} finals2000A.all (可选)"
    fi

    # 检查 DE440
    echo ""
    echo "检查 DE440 数据..."
    if [ -f "$INSTALL_DIR/de440/de440.bsp" ]; then
        echo -e "  ${GREEN}✓${NC} de440.bsp"
    else
        echo -e "  ${YELLOW}⚠${NC} de440.bsp (可选)"
    fi

    echo ""
    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}✓ 验证通过${NC}"
        return 0
    else
        echo -e "${RED}✗ 验证失败: $errors 个必需文件缺失${NC}"
        return 1
    fi
}

# 设置环境变量提示
print_env_hint() {
    echo ""
    echo "========================================"
    echo "环境变量配置"
    echo "========================================"
    echo ""
    echo "将以下行添加到 ~/.bashrc 或 ~/.zshrc:"
    echo ""
    echo -e "${BLUE}export OREKIT_DATA_DIR=${INSTALL_DIR}${NC}"
    echo ""
    echo "或使用项目本地路径:"
    echo ""
    echo -e "${BLUE}export OREKIT_DATA_DIR=/path/to/missionPlanAlgo/orekit-data${NC}"
}

# 打印安装摘要
print_summary() {
    echo ""
    echo "========================================"
    echo "安装摘要"
    echo "========================================"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}试运行完成，未实际安装文件${NC}"
        return
    fi

    echo "安装目录: $INSTALL_DIR"
    echo ""

    # 显示已安装文件
    echo "已安装 JAR 包:"
    ls -lh "$INSTALL_DIR/lib"/*.jar 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (无)"
    echo ""

    echo "数据文件:"
    if [ -f "$INSTALL_DIR/potential/egm-format/EGM2008_to2190_TideFree.gz" ]; then
        local egm_size=$(du -h "$INSTALL_DIR/potential/egm-format/EGM2008_to2190_TideFree.gz" | cut -f1)
        echo "  EGM2008: $egm_size"
    fi
    if [ -f "$INSTALL_DIR/IERS/finals2000A.all" ]; then
        local iers_size=$(du -h "$INSTALL_DIR/IERS/finals2000A.all" | cut -f1)
        echo "  IERS: $iers_size"
    fi
    if [ -f "$INSTALL_DIR/de440/de440.bsp" ]; then
        local de440_size=$(du -h "$INSTALL_DIR/de440/de440.bsp" | cut -f1)
        echo "  DE440: $de440_size"
    fi

    echo ""
    echo -e "${GREEN}安装完成!${NC}"
    echo ""
    echo "验证命令:"
    echo "  python -c \"from core.orbit.visibility.orekit_config import get_orekit_data_dir; print(get_orekit_data_dir())\""
}

# 主函数
main() {
    echo "========================================"
    echo "Orekit 数据文件下载脚本"
    echo "========================================"
    echo "安装目录: $INSTALL_DIR"
    echo "Orekit版本: $OREKIT_VERSION"
    echo "Hipparchus版本: $HIPPARCHUS_VERSION"
    if [ "$FORCE" = true ]; then
        echo -e "${YELLOW}模式: 强制重新下载${NC}"
    fi
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}模式: 试运行${NC}"
    fi
    echo "========================================"

    # 仅验证模式
    if [ "$VERIFY_ONLY" = true ]; then
        verify_installation
        exit $?
    fi

    # 检查依赖
    check_dependencies

    # 检查磁盘空间
    if [ "$DRY_RUN" = false ]; then
        check_disk_space
    fi

    # 创建目录
    create_directories

    # 下载 JAR 包
    download_orekit_jars

    # 下载 EGM2008（必须）
    if ! download_egm2008; then
        echo ""
        echo -e "${RED}EGM2008 数据下载失败，安装中止${NC}"
        exit 1
    fi

    # 下载 IERS
    download_iers

    # 下载 DE440
    download_de440

    # 验证安装
    verify_installation

    # 显示环境变量提示
    print_env_hint

    # 打印摘要
    print_summary
}

# 执行主函数
main
