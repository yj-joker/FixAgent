import importlib
import os

from test_runner import ask, print_json, run_auto_cases, run_menu


ENV_KEYS = [
    "DASHSCOPE_API_KEY",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "REDIS_HOST",
    "REDIS_PORT",
]


def reload_settings(env=None, clear=False):
    if clear:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
    if env:
        os.environ.update(env)
    import config.settings as settings_module
    return importlib.reload(settings_module)


def auto_test():
    def case_all_env():
        module = reload_settings({
            "DASHSCOPE_API_KEY": "sk-test",
            "LLM_MODEL": "qwen-plus",
            "LLM_TEMPERATURE": "0.2",
            "REDIS_HOST": "redis-test",
            "REDIS_PORT": "6380",
        }, clear=True)
        s = module.Settings()
        return {
            "dashscope_api_key": s.dashscope_api_key,
            "llm_model": s.llm_model,
            "llm_temperature": s.llm_temperature,
            "redis_host": s.redis_host,
            "redis_port": s.redis_port,
        }

    def case_defaults():
        module = reload_settings(clear=True)
        s = module.Settings()
        return {"llm_model": s.llm_model, "llm_temperature": s.llm_temperature}

    def case_type_convert():
        module = reload_settings({"LLM_TEMPERATURE": "0.7", "REDIS_PORT": "6379"}, clear=True)
        s = module.Settings()
        return {"llm_temperature_type": type(s.llm_temperature).__name__, "redis_port_type": type(s.redis_port).__name__}

    def case_singleton():
        module = reload_settings(clear=True)
        module._settings = None
        return module.get_settings() is module.get_settings()

    run_auto_cases([
        {
            "name": "所有环境变量正确设置时 Settings 各字段返回正确值",
            "input": "DASHSCOPE_API_KEY/LLM_MODEL/LLM_TEMPERATURE/REDIS_HOST/REDIS_PORT",
            "expected": {"dashscope_api_key": "sk-test", "redis_port": 6380},
            "run": case_all_env,
            "check": lambda x: x["dashscope_api_key"] == "sk-test" and x["redis_port"] == 6380,
        },
        {
            "name": ".env 文件或环境变量缺省时回退默认值",
            "input": "清空关键环境变量",
            "expected": {"llm_model": "qwen-plus", "llm_temperature": 0.7},
            "run": case_defaults,
            "check": lambda x: x == {"llm_model": "qwen-plus", "llm_temperature": 0.7},
        },
        {
            "name": "字符串环境变量正确转换为 float / int",
            "input": {"LLM_TEMPERATURE": "0.7", "REDIS_PORT": "6379"},
            "expected": {"llm_temperature_type": "float", "redis_port_type": "int"},
            "run": case_type_convert,
            "check": lambda x: x["llm_temperature_type"] == "float" and x["redis_port_type"] == "int",
        },
        {
            "name": "get_settings() 多次调用返回同一实例",
            "input": "连续调用 get_settings()",
            "expected": True,
            "run": case_singleton,
            "check": lambda x: x is True,
        },
    ])


def manual_test():
    module = reload_settings()
    s = module.get_settings()
    print("当前 Settings 实例字段:")
    print_json({k: getattr(s, k) for k in dir(s) if not k.startswith("_") and not callable(getattr(s, k))})
    key = ask("输入要临时查看的环境变量名", "LLM_MODEL")
    print(f"{key} = {os.getenv(key)}")


if __name__ == "__main__":
    run_menu("config/settings.py", auto_test, manual_test)
