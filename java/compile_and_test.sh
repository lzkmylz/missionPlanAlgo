#!/bin/bash

# Java编译和测试脚本
# 用于编译OptimizedVisibilityCalculator和运行测试

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Java编译和测试脚本 ===${NC}"

# 检查Java环境
if ! command -v javac &> /dev/null; then
    echo -e "${RED}错误: javac未找到。请安装JDK。${NC}"
    exit 1
fi

echo "Java版本:"
java -version 2>&1 | head -3

# 设置类路径
# 注意: 需要根据实际环境修改OREKIT_HOME
OREKIT_HOME="/usr/local/share/orekit"
CLASSPATH="classes"

# 尝试查找Orekit jar文件
if [ -d "$OREKIT_HOME" ]; then
    for jar in "$OREKIT_HOME"/*.jar; do
        if [ -f "$jar" ]; then
            CLASSPATH="$CLASSPATH:$jar"
        fi
    done
else
    echo -e "${YELLOW}警告: Orekit目录 $OREKIT_HOME 不存在${NC}"
    echo -e "${YELLOW}尝试使用现有编译的类...${NC}"
fi

echo ""
echo "类路径: $CLASSPATH"

# 编译Java源文件
echo ""
echo -e "${YELLOW}=== 编译Java源文件 ===${NC}"

# 编译OrbitStateCache.java
echo "编译 OrbitStateCache.java..."
javac -cp "$CLASSPATH" -d classes src/orekit/visibility/OrbitStateCache.java 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}编译 OrbitStateCache.java 失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ OrbitStateCache.java 编译成功${NC}"

# 编译OptimizedVisibilityCalculator.java
echo "编译 OptimizedVisibilityCalculator.java..."
javac -cp "$CLASSPATH" -d classes src/orekit/visibility/OptimizedVisibilityCalculator.java 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}编译 OptimizedVisibilityCalculator.java 失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ OptimizedVisibilityCalculator.java 编译成功${NC}"

# 编译测试类
echo "编译 OptimizedVisibilityCalculatorTest.java..."
javac -cp "$CLASSPATH" -d classes src/orekit/visibility/OptimizedVisibilityCalculatorTest.java 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}编译 OptimizedVisibilityCalculatorTest.java 失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ OptimizedVisibilityCalculatorTest.java 编译成功${NC}"

# 运行测试
echo ""
echo -e "${YELLOW}=== 运行测试 ===${NC}"
cd classes
java -cp ".:$CLASSPATH" orekit.visibility.OptimizedVisibilityCalculatorTest 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== 所有测试通过 ===${NC}"
else
    echo ""
    echo -e "${RED}=== 测试失败 ===${NC}"
    exit 1
fi
