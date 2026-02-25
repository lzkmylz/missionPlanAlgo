#!/bin/bash
#
# Orekit辅助类编译脚本
#
# 编译BatchStepHandler.java为orekit-helper.jar
#
# 使用方法:
#   ./compile_helper_jar.sh [安装目录]
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

# Orekit版本
OREKIT_VERSION="12.0"
HIPPARCHUS_VERSION="3.0"

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Java源文件路径
JAVA_SRC_DIR="$PROJECT_ROOT/java/src"
JAVA_CLASS_DIR="$PROJECT_ROOT/java/classes"
JAVA_FILE="$JAVA_SRC_DIR/orekit/helper/BatchStepHandler.java"

# 打印帮助信息
print_usage() {
    echo "用法: $0 [选项] [安装目录]"
    echo ""
    echo "选项:"
    echo "  -h, --help      显示帮助信息"
    echo "  -c, --clean     清理编译输出"
    echo ""
    echo "默认安装目录: /usr/local/share/orekit"
}

# 解析命令行参数
CLEAN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
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

# 检查Java环境
check_java() {
    echo "检查Java环境..."

    if ! command -v javac &> /dev/null; then
        echo -e "${RED}错误: 未找到javac，请安装JDK${NC}"
        exit 1
    fi

    if ! command -v jar &> /dev/null; then
        echo -e "${RED}错误: 未找到jar命令，请安装JDK${NC}"
        exit 1
    fi

    # 检查Java版本
    JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
    echo "Java版本: $JAVA_VERSION"

    # 提取主版本号
    JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f1)
    if [[ "$JAVA_MAJOR" -lt 17 ]]; then
        echo -e "${YELLOW}警告: Java版本过低，建议使用Java 17或更高版本${NC}"
    fi
}

# 检查源文件
check_source() {
    echo ""
    echo "检查Java源文件..."

    if [ ! -f "$JAVA_FILE" ]; then
        echo -e "${RED}错误: 源文件不存在: $JAVA_FILE${NC}"
        exit 1
    fi

    echo "源文件: $JAVA_FILE"
}

# 检查依赖jar
check_dependencies() {
    echo ""
    echo "检查依赖库..."

    local missing=()

    # Orekit jar
    OREKIT_JAR="$INSTALL_DIR/orekit-${OREKIT_VERSION}.jar"
    if [ ! -f "$OREKIT_JAR" ]; then
        missing+=("orekit-${OREKIT_VERSION}.jar")
    fi

    # Hipparchus jars
    HIPPARCHUS_JARS=(
        "$INSTALL_DIR/hipparchus-core-${HIPPARCHUS_VERSION}.jar"
        "$INSTALL_DIR/hipparchus-geometry-${HIPPARCHUS_VERSION}.jar"
        "$INSTALL_DIR/hipparchus-ode-${HIPPARCHUS_VERSION}.jar"
    )

    for jar in "${HIPPARCHUS_JARS[@]}"; do
        if [ ! -f "$jar" ]; then
            missing+=("$(basename "$jar")")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${YELLOW}警告: 缺少以下依赖jar文件:${NC}"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "请先运行 install_orekit_data.sh 安装依赖"
        exit 1
    fi

    echo -e "${GREEN}所有依赖已找到${NC}"
}

# 构建classpath
build_classpath() {
    echo ""
    echo "构建classpath..."

    CLASSPATH="$OREKIT_JAR"
    for jar in "${HIPPARCHUS_JARS[@]}"; do
        CLASSPATH="$CLASSPATH:$jar"
    done

    echo "Classpath: $CLASSPATH"
}

# 清理编译输出
clean() {
    echo ""
    echo "清理编译输出..."

    if [ -d "$JAVA_CLASS_DIR" ]; then
        rm -rf "$JAVA_CLASS_DIR"
        echo "已删除: $JAVA_CLASS_DIR"
    fi

    local output_jar="$INSTALL_DIR/orekit-helper.jar"
    if [ -f "$output_jar" ]; then
        rm -f "$output_jar"
        echo "已删除: $output_jar"
    fi

    echo -e "${GREEN}清理完成${NC}"
}

# 编译Java文件
compile() {
    echo ""
    echo "编译Java文件..."

    # 创建输出目录
    mkdir -p "$JAVA_CLASS_DIR"

    # 编译
    echo "执行: javac -cp \"$CLASSPATH\" -d \"$JAVA_CLASS_DIR\" \"$JAVA_FILE\""
    if ! javac -cp "$CLASSPATH" -d "$JAVA_CLASS_DIR" "$JAVA_FILE"; then
        echo -e "${RED}错误: 编译失败${NC}"
        exit 1
    fi

    echo -e "${GREEN}编译成功${NC}"
}

# 打包jar
package_jar() {
    echo ""
    echo "打包JAR文件..."

    local output_jar="$INSTALL_DIR/orekit-helper.jar"

    # 进入classes目录打包
    cd "$JAVA_CLASS_DIR"

    echo "执行: jar cf \"$output_jar\" orekit/helper/*.class"
    if ! jar cf "$output_jar" orekit/helper/*.class; then
        echo -e "${RED}错误: 打包失败${NC}"
        exit 1
    fi

    echo -e "${GREEN}打包成功: $output_jar${NC}"
}

# 验证jar
verify_jar() {
    echo ""
    echo "验证JAR文件..."

    local output_jar="$INSTALL_DIR/orekit-helper.jar"

    # 检查jar内容
    echo "JAR内容:"
    jar tf "$output_jar" | head -20

    # 验证类文件存在
    if jar tf "$output_jar" | grep -q "BatchStepHandler.class"; then
        echo -e "${GREEN}验证成功: BatchStepHandler.class 已包含${NC}"
    else
        echo -e "${RED}错误: BatchStepHandler.class 未找到${NC}"
        exit 1
    fi
}

# 打印摘要
print_summary() {
    echo ""
    echo "=========================================="
    echo "编译摘要"
    echo "=========================================="
    echo "源文件: $JAVA_FILE"
    echo "输出JAR: $INSTALL_DIR/orekit-helper.jar"
    echo ""
    echo "依赖库:"
    echo "  - orekit-${OREKIT_VERSION}.jar"
    echo "  - hipparchus-core-${HIPPARCHUS_VERSION}.jar"
    echo "  - hipparchus-geometry-${HIPPARCHUS_VERSION}.jar"
    echo "  - hipparchus-ode-${HIPPARCHUS_VERSION}.jar"
    echo ""
    echo -e "${GREEN}编译完成!${NC}"
    echo ""
    echo "使用方法:"
    echo "  在Python中通过JPype加载:"
    echo "    BatchStepHandler = jpype.JClass('orekit.helper.BatchStepHandler')"
}

# 主函数
main() {
    echo "=========================================="
    echo "Orekit辅助类编译脚本"
    echo "=========================================="
    echo "安装目录: $INSTALL_DIR"
    echo "=========================================="

    if [ "$CLEAN" = true ]; then
        check_java
        clean
        exit 0
    fi

    check_java
    check_source
    check_dependencies
    build_classpath
    compile
    package_jar
    verify_jar
    print_summary
}

# 执行主函数
main
