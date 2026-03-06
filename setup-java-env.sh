#!/bin/bash

# 设置Java环境变量脚本
# 将此脚本添加到 ~/.bashrc 或 ~/.profile 中以永久生效

# Java JDK路径 - 检测多个常见位置
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
        local java_path=$(readlink -f "$(command -v java)")
        # 通常是 /path/to/jdk/bin/java，所以取上两级目录
        local inferred_home="$(dirname "$(dirname "$java_path")")"
        if [ -d "$inferred_home" ]; then
            export JAVA_HOME="$inferred_home"
            return 0
        fi
    fi

    return 1
}

find_java_home

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
