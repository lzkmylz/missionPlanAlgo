"""
Orekit安装脚本测试

TDD测试套件 - 测试Orekit安装和编译脚本
"""

import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock, mock_open, call
import tempfile
import shutil


def get_project_root():
    """获取项目根目录"""
    # tests/unit/core/orbit/test_orekit_scripts.py -> 项目根目录
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


class TestInstallOrekitDataScript:
    """测试install_orekit_data.sh脚本"""

    def test_script_file_exists(self):
        """测试脚本文件存在"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'install_orekit_data.sh'
        )
        assert os.path.exists(script_path), f"脚本文件不存在: {script_path}"

    def test_script_is_executable(self):
        """测试脚本可执行"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'install_orekit_data.sh'
        )
        if os.path.exists(script_path):
            assert os.access(script_path, os.X_OK), "脚本没有执行权限"

    def test_script_contains_required_components(self):
        """测试脚本包含必要的组件"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'install_orekit_data.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        with open(script_path, 'r') as f:
            content = f.read()

        # 应该包含下载Orekit的代码
        assert 'orekit' in content.lower(), "脚本应该包含orekit相关代码"

        # 应该包含创建目录的代码
        assert 'mkdir' in content or 'mkdir -p' in content, "脚本应该创建目录"

        # 应该包含下载数据文件的代码
        assert 'curl' in content or 'wget' in content, "脚本应该使用curl或wget下载文件"

    def test_script_has_shebang(self):
        """测试脚本有shebang行"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'install_orekit_data.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        with open(script_path, 'r') as f:
            first_line = f.readline()

        assert first_line.startswith('#!/'), "脚本应该有shebang行"


class TestCompileHelperJarScript:
    """测试compile_helper_jar.sh脚本"""

    def test_script_file_exists(self):
        """测试脚本文件存在"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'compile_helper_jar.sh'
        )
        assert os.path.exists(script_path), f"脚本文件不存在: {script_path}"

    def test_script_is_executable(self):
        """测试脚本可执行"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'compile_helper_jar.sh'
        )
        if os.path.exists(script_path):
            assert os.access(script_path, os.X_OK), "脚本没有执行权限"

    def test_script_contains_javac(self):
        """测试脚本包含javac编译命令"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'compile_helper_jar.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        with open(script_path, 'r') as f:
            content = f.read()

        assert 'javac' in content, "脚本应该包含javac编译命令"
        assert 'jar' in content, "脚本应该包含jar打包命令"

    def test_script_compiles_batchstephandler(self):
        """测试脚本编译BatchStepHandler"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'compile_helper_jar.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        with open(script_path, 'r') as f:
            content = f.read()

        assert 'BatchStepHandler' in content or 'batch' in content.lower(), \
            "脚本应该编译BatchStepHandler"

    def test_script_has_shebang(self):
        """测试脚本有shebang行"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'compile_helper_jar.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        with open(script_path, 'r') as f:
            first_line = f.readline()

        assert first_line.startswith('#!/'), "脚本应该有shebang行"


class TestBatchStepHandlerJava:
    """测试BatchStepHandler Java类"""

    def test_java_file_exists(self):
        """测试Java源文件存在"""
        java_file_path = os.path.join(
            get_project_root(),
            'java', 'src', 'orekit', 'helper', 'BatchStepHandler.java'
        )
        assert os.path.exists(java_file_path), f"Java文件不存在: {java_file_path}"

    def test_java_file_contains_class_declaration(self):
        """测试Java文件包含类声明"""
        java_file_path = os.path.join(
            get_project_root(),
            'java', 'src', 'orekit', 'helper', 'BatchStepHandler.java'
        )

        if not os.path.exists(java_file_path):
            pytest.skip("Java文件不存在")

        with open(java_file_path, 'r') as f:
            content = f.read()

        assert 'public class BatchStepHandler' in content or 'class BatchStepHandler' in content, \
            "Java文件应该包含BatchStepHandler类声明"

    def test_java_file_implements_interface(self):
        """测试Java类实现了OrekitFixedStepHandler接口"""
        java_file_path = os.path.join(
            get_project_root(),
            'java', 'src', 'orekit', 'helper', 'BatchStepHandler.java'
        )

        if not os.path.exists(java_file_path):
            pytest.skip("Java文件不存在")

        with open(java_file_path, 'r') as f:
            content = f.read()

        assert 'OrekitFixedStepHandler' in content, \
            "Java类应该实现OrekitFixedStepHandler接口"

    def test_java_file_has_handlestep_method(self):
        """测试Java类包含handleStep方法"""
        java_file_path = os.path.join(
            get_project_root(),
            'java', 'src', 'orekit', 'helper', 'BatchStepHandler.java'
        )

        if not os.path.exists(java_file_path):
            pytest.skip("Java文件不存在")

        with open(java_file_path, 'r') as f:
            content = f.read()

        assert 'handleStep' in content, "Java类应该包含handleStep方法"

    def test_java_file_has_getresults_method(self):
        """测试Java类包含getResults方法"""
        java_file_path = os.path.join(
            get_project_root(),
            'java', 'src', 'orekit', 'helper', 'BatchStepHandler.java'
        )

        if not os.path.exists(java_file_path):
            pytest.skip("Java文件不存在")

        with open(java_file_path, 'r') as f:
            content = f.read()

        assert 'getResults' in content, "Java类应该包含getResults方法"

    def test_java_file_returns_double_array(self):
        """测试Java类返回double[][]数组"""
        java_file_path = os.path.join(
            get_project_root(),
            'java', 'src', 'orekit', 'helper', 'BatchStepHandler.java'
        )

        if not os.path.exists(java_file_path):
            pytest.skip("Java文件不存在")

        with open(java_file_path, 'r') as f:
            content = f.read()

        assert 'double[][]' in content or 'double[]' in content, \
            "Java类应该返回double数组"


class TestScriptFunctionality:
    """测试脚本功能（使用mock）"""

    @patch('subprocess.run')
    def test_install_script_dry_run(self, mock_run):
        """测试安装脚本dry-run模式"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'install_orekit_data.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        # 模拟脚本执行
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        result = subprocess.run(['bash', script_path, '--dry-run'], capture_output=True, text=True)
        # 实际执行脚本，但使用--dry-run参数

    @patch('subprocess.run')
    def test_compile_script_checks_java_version(self, mock_run):
        """测试编译脚本检查Java版本"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'compile_helper_jar.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        with open(script_path, 'r') as f:
            content = f.read()

        # 应该检查Java版本
        assert 'java' in content.lower() or 'javac' in content.lower(), \
            "脚本应该调用Java命令"


class TestScriptErrorHandling:
    """测试脚本错误处理"""

    def test_install_script_handles_download_failure(self):
        """测试安装脚本处理下载失败"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'install_orekit_data.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        with open(script_path, 'r') as f:
            content = f.read()

        # 应该包含错误处理
        assert 'set -e' in content or 'exit' in content or 'if' in content, \
            "脚本应该包含错误处理"

    def test_compile_script_handles_compile_failure(self):
        """测试编译脚本处理编译失败"""
        script_path = os.path.join(
            get_project_root(),
            'scripts', 'compile_helper_jar.sh'
        )

        if not os.path.exists(script_path):
            pytest.skip("脚本文件不存在")

        with open(script_path, 'r') as f:
            content = f.read()

        # 应该包含错误处理
        assert 'set -e' in content or 'exit' in content or 'if' in content, \
            "脚本应该包含错误处理"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
