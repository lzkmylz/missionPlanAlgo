# Makefile for OptimizedVisibilityCalculator

# Java配置
JAVA_HOME ?= $(HOME)/jdk-17.0.9+9
JAVA := $(JAVA_HOME)/bin/java
JAVAC := $(JAVA_HOME)/bin/javac

# 类路径
CP := classes
CP := $(CP):lib/orekit-12.0.jar
CP := $(CP):lib/hipparchus-core-3.0.jar
CP := $(CP):lib/hipparchus-geometry-3.0.jar
CP := $(CP):lib/hipparchus-ode-3.0.jar
CP := $(CP):lib/json.jar

# Parquet依赖
CP := $(CP):lib/parquet-common-1.13.1.jar
CP := $(CP):lib/parquet-format-structures-1.13.1.jar
CP := $(CP):lib/parquet-column-1.13.1.jar
CP := $(CP):lib/parquet-encoding-1.13.1.jar
CP := $(CP):lib/parquet-hadoop-1.13.1.jar
CP := $(CP):lib/parquet-jackson-1.13.1.jar
CP := $(CP):lib/hadoop-common-3.3.6.jar
CP := $(CP):lib/hadoop-mapreduce-client-core-3.3.6.jar
CP := $(CP):lib/avro-1.11.3.jar
CP := $(CP):lib/jackson-core-2.15.2.jar
CP := $(CP):lib/jackson-databind-2.15.2.jar
CP := $(CP):lib/jackson-annotations-2.15.2.jar
CP := $(CP):lib/slf4j-api-2.0.9.jar
CP := $(CP):lib/slf4j-simple-2.0.9.jar
CP := $(CP):lib/commons-lang3-3.12.0.jar
CP := $(CP):lib/guava-32.1.3-jre.jar
CP := $(CP):lib/woodstox-core-6.5.1.jar
CP := $(CP):lib/stax2-api-4.2.1.jar
CP := $(CP):lib/commons-collections-3.2.2.jar

# 源代码目录
SRC_DIR := src/orekit/visibility
MODEL_DIR := $(SRC_DIR)/model

# 颜色
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: all build test clean rebuild help

# 默认目标
all: build

# 显示帮助
help:
	@echo "$(BLUE)OptimizedVisibilityCalculator Makefile$(NC)"
	@echo ""
	@echo "可用命令:"
	@echo "  $(GREEN)make build$(NC)   - 编译所有Java源文件"
	@echo "  $(GREEN)make test$(NC)    - 运行测试"
	@echo "  $(GREEN)make rebuild$(NC) - 清理并重新编译"
	@echo "  $(GREEN)make clean$(NC)   - 清理编译结果"
	@echo "  $(GREEN)make help$(NC)    - 显示此帮助"

# 编译所有文件
build: compile-model compile-cache compile-exporter compile-main compile-test
	@echo "$(GREEN)所有文件编译成功!$(NC)"

# 编译模型类
compile-model:
	@echo "$(YELLOW)编译模型类...$(NC)"
	@mkdir -p classes
	$(JAVAC) -cp "$(CP)" -d classes $(MODEL_DIR)/*.java

# 编译JsonScenarioLoader（需要在OrbitStateCache之前，因为OrbitStateCache依赖它）
compile-loader: compile-model
	@echo "$(YELLOW)编译 JsonScenarioLoader.java...$(NC)"
	$(JAVAC) -cp "$(CP)" -d classes $(SRC_DIR)/JsonScenarioLoader.java

# 编译SmartOrbitInitializer（需要在OrbitStateCache之前）
compile-initializer: compile-loader
	@echo "$(YELLOW)编译 SmartOrbitInitializer.java...$(NC)"
	$(JAVAC) -cp "$(CP)" -d classes $(SRC_DIR)/SmartOrbitInitializer.java

# 编译AttitudeCalculator（需要在OptimizedVisibilityCalculator之前）
compile-attitude: compile-initializer
	@echo "$(YELLOW)编译 AttitudeCalculator.java...$(NC)"
	$(JAVAC) -cp "$(CP)" -d classes $(SRC_DIR)/AttitudeCalculator.java

# 编译OrbitStateCache
compile-cache: compile-attitude
	@echo "$(YELLOW)编译 OrbitStateCache.java...$(NC)"
	$(JAVAC) -cp "$(CP)" -d classes $(SRC_DIR)/OrbitStateCache.java

# 编译OrbitDataExporter（依赖OrbitStateCache）
compile-exporter: compile-cache
	@echo "$(YELLOW)编译 OrbitDataExporter.java...$(NC)"
	$(JAVAC) -cp "$(CP)" -d classes $(SRC_DIR)/OrbitDataExporter.java

# 编译OptimizedVisibilityCalculator
compile-calculator: compile-cache
	@echo "$(YELLOW)编译 OptimizedVisibilityCalculator.java...$(NC)"
	$(JAVAC) -cp "$(CP)" -d classes $(SRC_DIR)/OptimizedVisibilityCalculator.java

# 编译LargeScaleFrequencyTest（依赖OrbitDataExporter）
compile-main: compile-exporter compile-calculator
	@echo "$(YELLOW)编译 LargeScaleFrequencyTest.java...$(NC)"
	$(JAVAC) -cp "$(CP)" -d classes $(SRC_DIR)/LargeScaleFrequencyTest.java

# 编译测试类
compile-test: compile-calculator
	@echo "$(YELLOW)编译测试类...$(NC)"
	$(JAVAC) -cp "$(CP)" -d classes $(SRC_DIR)/OptimizedVisibilityCalculatorTest.java

# 运行测试
test: build
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE)  运行 OptimizedVisibilityCalculator 测试$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo ""
	$(JAVA) -cp "$(CP)" orekit.visibility.OptimizedVisibilityCalculatorTest

# 清理编译结果
clean:
	@echo "$(YELLOW)清理编译结果...$(NC)"
	@rm -rf classes/*.class classes/orekit classes/org
	@echo "$(GREEN)清理完成$(NC)"

# 重新编译
rebuild: clean build

# 快速测试（不重新编译）
quick-test:
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE)  运行测试（使用现有编译文件）$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo ""
	$(JAVA) -cp "$(CP)" orekit.visibility.OptimizedVisibilityCalculatorTest
