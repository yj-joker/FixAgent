"""pytest 公共夹具：在收集测试前安装可选依赖 stub。

manual_tests 下的脚本原本通过 test_runner._install_optional_dependency_stubs()
在导入业务模块前打桩（redis / neo4j / dashscope / httpx / fastapi 等未安装时）。
pytest 收集阶段不会执行各脚本里的运行器逻辑，因此在此 conftest 中提前调用，
保证 import services.* 能在缺少重依赖的环境下成功。
"""
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_runner_path = Path(__file__).resolve().parent / "test_runner.py"
_spec = importlib.util.spec_from_file_location("_mt_test_runner", _runner_path)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)

_runner._install_optional_dependency_stubs()
