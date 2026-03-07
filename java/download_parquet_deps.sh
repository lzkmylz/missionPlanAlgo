#!/bin/bash
# 下载Parquet依赖库

set -e

LIB_DIR="$(dirname "$0")/lib"
mkdir -p "$LIB_DIR"

echo "下载Parquet依赖库..."
cd "$LIB_DIR"

# Maven Central URL
MVN_URL="https://repo1.maven.org/maven2"

# 定义依赖
# Parquet 1.13.1 版本与Orekit 12.0兼容性好
declare -a DEPS=(
    # Parquet核心库
    "org/apache/parquet/parquet-common/1.13.1/parquet-common-1.13.1.jar"
    "org/apache/parquet/parquet-format-structures/1.13.1/parquet-format-structures-1.13.1.jar"
    "org/apache/parquet/parquet-column/1.13.1/parquet-column-1.13.1.jar"
    "org/apache/parquet/parquet-encoding/1.13.1/parquet-encoding-1.13.1.jar"
    "org/apache/parquet/parquet-hadoop/1.13.1/parquet-hadoop-1.13.1.jar"
    "org/apache/parquet/parquet-jackson/1.13.1/parquet-jackson-1.13.1.jar"

    # Hadoop依赖（Parquet需要）
    "org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar"
    "org/apache/hadoop/hadoop-mapreduce-client-core/3.3.6/hadoop-mapreduce-client-core-3.3.6.jar"

    # Avro（Parquet可选依赖，用于schema）
    "org/apache/avro/avro/1.11.3/avro-1.11.3.jar"

    # Jackson（JSON处理）
    "com/fasterxml/jackson/core/jackson-core/2.15.2/jackson-core-2.15.2.jar"
    "com/fasterxml/jackson/core/jackson-databind/2.15.2/jackson-databind-2.15.2.jar"
    "com/fasterxml/jackson/core/jackson-annotations/2.15.2/jackson-annotations-2.15.2.jar"

    # SLF4J（日志）
    "org/slf4j/slf4j-api/2.0.9/slf4j-api-2.0.9.jar"
    "org/slf4j/slf4j-simple/2.0.9/slf4j-simple-2.0.9.jar"

    # Apache Commons
    "org/apache/commons/commons-lang3/3.12.0/commons-lang3-3.12.0.jar"

    # Guava（Hadoop需要）
    "com/google/guava/guava/32.1.3-jre/guava-32.1.3-jre.jar"
)

# 下载每个依赖
for dep in "${DEPS[@]}"; do
    filename=$(basename "$dep")
    if [ -f "$filename" ]; then
        echo "  已存在: $filename"
    else
        echo "  下载: $filename"
        curl -L -o "$filename" "$MVN_URL/$dep" || {
            echo "    警告: 下载失败 $filename"
        }
    fi
done

echo "Parquet依赖库下载完成"
echo ""
echo "下载的库文件:"
ls -lh *.jar | awk '{print "  " $9 " (" $5 ")"}'
