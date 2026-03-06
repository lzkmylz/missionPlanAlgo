#!/bin/bash

# 设置Java环境变量脚本
# 将此脚本添加到 ~/.bashrc 或 ~/.profile 中以永久生效

# Java JDK路径
if [ -d "$HOME/jdk-17.0.9+9" ]; then
    export JAVA_HOME="$HOME/jdk-17.0.9+9"
elif [ -d "/usr/lib/jvm/java-17" ]; then
    export JAVA_HOME="/usr/lib/jvm/java-17"
fi

# 添加Java到PATH
if [ -n "$JAVA_HOME" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "Java环境已设置:"
    echo "  JAVA_HOME = $JAVA_HOME"
    java -version 2>&1 | head -3
else
    echo "警告: 未找到Java JDK"
    echo "请安装Java 17或设置JAVA_HOME环境变量"
fi

# Orekit数据路径（可选）
if [ -d "$HOME/orekit-data" ]; then
    export OREKIT_DATA_PATH="$HOME/orekit-data"
    echo "  OREKIT_DATA_PATH = $OREKIT_DATA_PATH"
fi

# 添加自定义命令到PATH
if [ -d "$HOME/.local/bin" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "快捷命令:"
echo "  sat-visibility-test      - 运行可见性测试"
echo "  sat-visibility-test -r   - 重新编译并运行测试"
