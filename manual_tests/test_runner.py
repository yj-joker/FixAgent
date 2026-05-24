"""
FixAgent 手动测试脚本公共运行器。

每个测试脚本都可以直接用 `python manual_tests/test_xxx.py` 运行。
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def _install_optional_dependency_stubs() -> None:
    """自动测试允许在未安装外部依赖时导入业务模块，真实手动测试仍建议用项目 venv。"""
    def real_module_available(name: str) -> bool:
        if name in sys.modules:
            return True
        try:
            importlib.import_module(name)
            return True
        except ImportError:
            return False

    if not real_module_available("redis"):
        redis_mod = types.ModuleType("redis")

        class ResponseError(Exception):
            pass

        class Redis:
            def __init__(self, *args, **kwargs):
                self.storage = {}

            def get(self, key):
                return self.storage.get(key)

            def setex(self, key, ttl, value):
                self.storage[key] = value
                return True

            def hset(self, key, mapping=None):
                self.storage[key] = mapping or {}
                return 1

            def delete(self, key):
                return 1 if self.storage.pop(key, None) is not None else 0

            def execute_command(self, *args, **kwargs):
                return [0]

        redis_mod.Redis = Redis
        redis_mod.exceptions = types.SimpleNamespace(ResponseError=ResponseError)
        sys.modules["redis"] = redis_mod

    if not real_module_available("dashscope"):
        dashscope_mod = types.ModuleType("dashscope")
        dashscope_mod.api_key = None

        class MultiModalEmbedding:
            @staticmethod
            def call(*args, **kwargs):
                return types.SimpleNamespace(
                    status_code=200,
                    message="ok",
                    output={"embeddings": [{"index": 0, "embedding": [0.0] * 1024}]},
                )

        dashscope_mod.MultiModalEmbedding = MultiModalEmbedding
        sys.modules["dashscope"] = dashscope_mod

    if not real_module_available("neo4j"):
        neo4j_mod = types.ModuleType("neo4j")

        class GraphDatabase:
            @staticmethod
            def driver(*args, **kwargs):
                return types.SimpleNamespace(
                    session=lambda *a, **k: types.SimpleNamespace(
                        __enter__=lambda self: self,
                        __exit__=lambda self, exc_type, exc, tb: None,
                        run=lambda *ra, **rk: [],
                    ),
                    close=lambda: None,
                )

        neo4j_mod.GraphDatabase = GraphDatabase
        sys.modules["neo4j"] = neo4j_mod

    if not real_module_available("httpx"):
        httpx_mod = types.ModuleType("httpx")

        class Limits:
            def __init__(self, *args, **kwargs):
                pass

        class AsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def post(self, *args, **kwargs):
                raise RuntimeError("httpx 未安装，无法执行真实网络请求")

        httpx_mod.Limits = Limits
        httpx_mod.AsyncClient = AsyncClient
        sys.modules["httpx"] = httpx_mod

    if not real_module_available("fastapi"):
        fastapi_mod = types.ModuleType("fastapi")

        class HTTPException(Exception):
            def __init__(self, status_code=500, detail=None):
                self.status_code = status_code
                self.detail = detail
                super().__init__(detail)

        class FastAPI:
            def __init__(self, *args, **kwargs):
                self.routes = []

            def add_middleware(self, *args, **kwargs):
                return None

            def mount(self, *args, **kwargs):
                return None

            def post(self, *args, **kwargs):
                def decorator(func):
                    self.routes.append(("POST", args, kwargs, func))
                    return func
                return decorator

            get = post
            delete = post

            def exception_handler(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

        fastapi_mod.FastAPI = FastAPI
        fastapi_mod.HTTPException = HTTPException
        sys.modules["fastapi"] = fastapi_mod

        responses_mod = types.ModuleType("fastapi.responses")

        class JSONResponse:
            def __init__(self, content=None, status_code=200, **kwargs):
                self.content = content
                self.status_code = status_code

        class StreamingResponse:
            def __init__(self, content, media_type=None, headers=None):
                self.body_iterator = content
                self.media_type = media_type
                self.headers = headers or {}

        responses_mod.JSONResponse = JSONResponse
        responses_mod.StreamingResponse = StreamingResponse
        sys.modules["fastapi.responses"] = responses_mod

        cors_mod = types.ModuleType("fastapi.middleware.cors")

        class CORSMiddleware:
            pass

        cors_mod.CORSMiddleware = CORSMiddleware
        sys.modules["fastapi.middleware.cors"] = cors_mod

        staticfiles_mod = types.ModuleType("fastapi.staticfiles")

        class StaticFiles:
            def __init__(self, *args, **kwargs):
                pass

        staticfiles_mod.StaticFiles = StaticFiles
        sys.modules["fastapi.staticfiles"] = staticfiles_mod


_install_optional_dependency_stubs()


def print_header(module_path: str) -> None:
    print("=" * 40)
    print("  FixAgent 模块测试工具")
    print(f"  当前模块: {module_path}")
    print("=" * 40)
    print("请选择测试模式:")
    print("  1. 自动测试 - 使用预设数据批量运行")
    print("  2. 手动测试 - 手动输入数据逐项验证")
    print("  3. 退出")
    print("=" * 40)


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    return str(value)


def print_json(value: Any) -> None:
    try:
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    except TypeError:
        print(value)


def require_real_dependency(module_name: str, install_hint: str) -> Any:
    module = importlib.import_module(module_name)
    if getattr(module, "__file__", None) is None:
        raise RuntimeError(
            f"{module_name} 当前是测试 stub，不是真实依赖。请先安装真实包后再跑手动真实环境测试：{install_hint}"
        )
    return module


def require_env_value(name: str, hint: str) -> str:
    import os

    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}。{hint}")
    return value


def run_auto_cases(cases: list[dict[str, Any]]) -> None:
    passed = 0
    failed = 0
    total = len(cases)

    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{total}] {case['name']}")
        print(f"  输入: {format_value(case.get('input', '(无)'))}")
        print(f"  预期: {format_value(case.get('expected', '(见验证点)'))}")

        try:
            actual = case["run"]()
            checker: Callable[[Any], bool] = case.get("check", bool)
            ok = checker(actual)
        except Exception as exc:
            actual = f"{type(exc).__name__}: {exc}"
            ok = bool(case.get("expect_exception") and case["expect_exception"] in actual)

        print(f"  实际: {format_value(actual)}")
        if ok:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL")
            failed += 1

    print("\n" + "=" * 40)
    print(f"测试完成: 通过 {passed} / 失败 {failed} / 总计 {total}")
    print("=" * 40)


def run_async(coro):
    return asyncio.run(coro)


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f"（默认: {default}）" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else (default or "")


def ask_int(prompt: str, default: int) -> int:
    raw = ask(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        print("输入不是有效整数，已使用默认值。")
        return default


def run_menu(module_path: str, auto_func: Callable[[], None], manual_func: Callable[[], None]) -> None:
    while True:
        print_header(module_path)
        choice = input("请输入数字 (1/2/3): ").strip()
        if choice == "1":
            try:
                auto_func()
            except Exception as exc:
                print(f"自动测试执行异常: {type(exc).__name__}: {exc}")
        elif choice == "2":
            try:
                manual_func()
            except Exception as exc:
                print(f"手动测试执行异常: {type(exc).__name__}: {exc}")
        elif choice == "3":
            print("测试结束，再见！")
            return
        else:
            print("输入无效，请重新选择")


def patch_attr(obj: Any, name: str, value: Any):
    original = getattr(obj, name)
    setattr(obj, name, value)

    class Restore:
        def __enter__(self):
            return value

        def __exit__(self, exc_type, exc, tb):
            setattr(obj, name, original)

    return Restore()
