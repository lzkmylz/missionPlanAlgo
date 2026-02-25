#!/bin/bash
# Java 17 安装脚本 - 使用预编译版本

set -e

echo "=== Java 17 安装脚本 ==="
echo ""

# 创建临时目录
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# 检测架构
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    # M1/M2 Mac
    JDK_URL="https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.9%2B9/OpenJDK17U-jdk_aarch64_mac_hotspot_17.0.9_9.tar.gz"
    JDK_FILE="OpenJDK17U-jdk_aarch64_mac_hotspot_17.0.9_9.tar.gz"
else
    # Intel Mac
    JDK_URL="https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.9%2B9/OpenJDK17U-jdk_x64_mac_hotspot_17.0.9_9.tar.gz"
    JDK_FILE="OpenJDK17U-jdk_x64_mac_hotspot_17.0.9_9.tar.gz"
fi

echo "[1/5] 下载 Eclipse Temurin JDK 17..."
echo "架构: $ARCH"
curl -L -o "$JDK_FILE" "$JDK_URL"

echo "[2/5] 解压 JDK..."
tar -xzf "$JDK_FILE"

echo "[3/5] 安装到系统目录..."
sudo mv jdk-17.0.9+9/Contents/Home /Library/Java/JavaVirtualMachines/temurin-17.jdk

# 创建 Info.plist 使系统识别
sudo mkdir -p /Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents
cat | sudo tee /Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Info.plist > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Eclipse Temurin 17</string>
    <key>CFBundleIdentifier</key>
    <string>net.temurin.17.jdk</string>
    <key>CFBundleVersion</key>
    <string>17.0.9</string>
    <key>JVMMinimumFrameworkVersion</key>
    <string>13</string>
    <key>JVMMaximumFrameworkVersion</key>
    <string>999</string>
    <key>JVMMinimumSystemVersion</key>
    <string>10.6</string>
    <key>JVMCapabilities</key>
    <array>
        <string>CommandLine</string>
        <string>BundledApp</string>
    </array>
</dict>
</plist>
EOF

echo "[4/5] 配置环境变量..."
SHELL_RC="$HOME/.zshrc"
if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bash_profile"
fi

# 检查是否已配置
if ! grep -q "JAVA_HOME.*temurin-17" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# Java 17 Configuration" >> "$SHELL_RC"
    echo 'export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home' >> "$SHELL_RC"
    echo 'export PATH=$JAVA_HOME/bin:$PATH' >> "$SHELL_RC"
    echo "✅ 环境变量已添加到 $SHELL_RC"
else
    echo "ℹ️  环境变量已存在"
fi

# 立即生效
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

echo "[5/5] 验证安装..."
if java -version 2>&1 | grep -q "17"; then
    echo ""
    echo "✅ Java 17 安装成功！"
    echo ""
    java -version 2>&1
    echo ""
    echo "请运行以下命令使配置永久生效："
    echo "  source $SHELL_RC"
else
    echo "❌ 安装验证失败"
    exit 1
fi

# 清理
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "=== 安装完成 ==="
