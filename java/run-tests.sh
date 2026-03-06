#!/bin/bash

# OptimizedVisibilityCalculator 测试运行脚本
# 自动设置Java环境并运行测试

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 查找Java安装
find_java_home() {
    # 如果已经设置了 JAVA_HOME，直接使用
    if [ -n "$JAVA_HOME" ] && [ -d "$JAVA_HOME" ]; then
        return 0
    fi

    # 检测常见安装位置
    local java_paths=(
        "$HOME/jdk-17.0.9+9"
        "$HOME/jdk-17"
        "$HOME/jdk-21"
        "/usr/lib/jvm/java-17"
        "/usr/lib/jvm/java-17-openjdk"
        "/usr/lib/jvm/java-17-openjdk-amd64"
        "/usr/lib/jvm/java-21"
        "/usr/lib/jvm/java-21-openjdk"
        "/usr/lib/jvm/java-21-openjdk-amd64"
        "/opt/jdk-17"
        "/opt/jdk-21"
    )

    for path in "${java_paths[@]}"; do
        if [ -d "$path" ]; then
            export JAVA_HOME="$path"
            return 0
        fi
    done

    # 尝试从 which java 推断
    if command -v java &> /dev/null; then
        local java_path=$(readlink -f "$(command -v java)" 2>/dev/null || command -v java)
        # 通常是 /path/to/jdk/bin/java，所以取上两级目录
        if [[ "$java_path" == */bin/java ]]; then
            local inferred_home="$(dirname "$(dirname "$java_path")")"
            if [ -d "$inferred_home" ]; then
                export JAVA_HOME="$inferred_home"
                return 0
            fi
        fi
    fi

    return 1
}

# 设置Java环境
find_java_home

if [ -n "$JAVA_HOME" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
else
    echo -e "${RED}错误: 找不到Java JDK${NC}"
    echo "请安装Java 17+ 或设置 JAVA_HOME 环境变量"
    echo "常见安装位置:"
    echo "  - ~/jdk-17.0.9+9"
    echo "  - /usr/lib/jvm/java-17"
    echo "  - /usr/lib/jvm/java-17-openjdk"
    exit 1
fi

# 设置类路径
CP="classes"
for jar in lib/*.jar; do
    if [ -f "$jar" ]; then
        CP="$CP:$jar"
    fi
done

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  OptimizedVisibilityCalculator 测试运行器${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${YELLOW}Java版本:${NC}"
java -version 2>&1 | head -3
echo ""
echo -e "${YELLOW}类路径:${NC} $CP"
echo ""

# 检查是否需要重新编译
if [ "$1" == "--rebuild" ] || [ "$1" == "-r" ]; then
    echo -e "${YELLOW}重新编译Java源文件...${NC}"

    # 编译模型类
    echo "编译模型类..."
    javac -cp "$CP" -d classes src/orekit/visibility/model/*.java 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}模型类编译失败${NC}"
        exit 1
    fi

    # 编译主类
    echo "编译 OrbitStateCache.java..."
    javac -cp "$CP" -d classes src/orekit/visibility/OrbitStateCache.java 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}OrbitStateCache编译失败${NC}"
        exit 1
    fi

    echo "编译 OptimizedVisibilityCalculator.java..."
    javac -cp "$CP" -d classes src/orekit/visibility/OptimizedVisibilityCalculator.java 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}OptimizedVisibilityCalculator编译失败${NC}"
        exit 1
    fi

    echo "编译测试类..."
    javac -cp "$CP" -d classes src/orekit/visibility/OptimizedVisibilityCalculatorTest.java 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}测试类编译失败${NC}"
        exit 1
    fi

    echo -e "${GREEN}编译成功!${NC}"
    echo ""
fi

# 运行测试
echo -e "${YELLOW}运行测试...${NC}"
echo ""
java -cp "$CP" orekit.visibility.OptimizedVisibilityCalculatorTest 2>&1

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}  所有测试通过!${NC}"
    echo -e "${GREEN}================================================${NC}"
else
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}  测试失败${NC}"
    echo -e "${RED}================================================${NC}"
fi

exit $EXIT_CODE
