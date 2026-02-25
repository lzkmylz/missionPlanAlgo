#!/bin/bash
# 卸载 Java 9 脚本

set -e

echo "=== Java 9 卸载脚本 ==="
echo ""

# 检查 Java 9 是否存在
if [ ! -d "/Library/Java/JavaVirtualMachines/jdk-9.0.4.jdk" ]; then
    echo "❌ 未找到 Java 9 安装"
    exit 1
fi

echo "[1/4] 备份当前配置..."
current_java=$(java -version 2>&1 | head -1)
echo "卸载前版本: $current_java" > ~/.java_uninstall_backup.txt
echo "卸载时间: $(date)" >> ~/.java_uninstall_backup.txt

echo "[2/4] 删除 Java 9 JDK..."
sudo rm -rf /Library/Java/JavaVirtualMachines/jdk-9.0.4.jdk
echo "✅ JDK 目录已删除"

echo "[3/4] 清理 Java 插件..."
if [ -d "/Library/Internet Plug-Ins/JavaAppletPlugin.plugin" ]; then
    sudo rm -rf "/Library/Internet Plug-Ins/JavaAppletPlugin.plugin"
    echo "✅ 插件已删除"
else
    echo "ℹ️  无插件需要清理"
fi

if [ -d "/Library/PreferencePanes/JavaControlPanel.prefPane" ]; then
    sudo rm -rf "/Library/PreferencePanes/JavaControlPanel.prefPane"
    echo "✅ 控制面板已删除"
else
    echo "ℹ️  无控制面板需要清理"
fi

echo "[4/4] 验证卸载..."
if [ -d "/Library/Java/JavaVirtualMachines/jdk-9.0.4.jdk" ]; then
    echo "❌ 卸载失败，目录仍然存在"
    exit 1
else
    echo "✅ Java 9 已成功卸载"
fi

echo ""
echo "=== 卸载完成 ==="
echo ""
echo "当前 Java 版本:"
java -version 2>&1 || echo "系统中无其他 Java 版本"
echo ""
echo "如需安装 Java 17，请运行:"
echo "  brew install openjdk@17"
