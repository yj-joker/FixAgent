from test_runner import print_json, run_async, run_auto_cases, run_menu


def auto_test():
    from tools.base_tool import BaseTool, ToolException

    class DemoTool(BaseTool):
        mode = "ok"

        @property
        def name(self):
            return "demo_tool"

        @property
        def description(self):
            return "演示工具"

        async def _execute(self, **kwargs):
            if self.mode == "tool_error":
                raise ToolException("MY_ERR", "业务错误")
            if self.mode == "runtime":
                raise RuntimeError("未知错误")
            return {"echo": kwargs}

    async def run_ok():
        tool = DemoTool()
        return (await tool.run(value=1)).model_dump()

    async def run_tool_exception():
        tool = DemoTool()
        tool.mode = "tool_error"
        return (await tool.run()).model_dump()

    async def run_unknown_exception():
        tool = DemoTool()
        tool.mode = "runtime"
        return (await tool.run()).model_dump()

    run_auto_cases([
        {
            "name": "run() 正常返回 ToolResult(success=True)",
            "input": {"value": 1},
            "expected": {"success": True, "tool_name": "demo_tool"},
            "run": lambda: run_async(run_ok()),
            "check": lambda x: x["success"] is True and x["data"]["echo"]["value"] == 1,
        },
        {
            "name": "ToolException 被捕获为结构化错误",
            "input": "raise ToolException('MY_ERR')",
            "expected": {"success": False, "error.code": "MY_ERR"},
            "run": lambda: run_async(run_tool_exception()),
            "check": lambda x: x["success"] is False and x["error"]["code"] == "MY_ERR",
        },
        {
            "name": "未知异常被捕获为 TOOL_ERROR",
            "input": "raise RuntimeError",
            "expected": {"error.code": "TOOL_ERROR"},
            "run": lambda: run_async(run_unknown_exception()),
            "check": lambda x: x["error"]["code"] == "TOOL_ERROR",
        },
        {
            "name": "to_openai_schema() 返回 function schema",
            "input": "DemoTool().to_openai_schema()",
            "expected": "function.name=demo_tool",
            "run": lambda: DemoTool().to_openai_schema(),
            "check": lambda x: x["type"] == "function" and x["function"]["name"] == "demo_tool",
        },
        {
            "name": "get_parameters_schema() 默认空对象",
            "input": "DemoTool().get_parameters_schema()",
            "expected": {"type": "object", "properties": {}, "required": []},
            "run": lambda: DemoTool().get_parameters_schema(),
            "check": lambda x: x == {"type": "object", "properties": {}, "required": []},
        },
    ])


def manual_test():
    from tools.base_tool import BaseTool

    class DemoTool(BaseTool):
        @property
        def name(self):
            return "demo_tool"

        @property
        def description(self):
            return "演示工具"

        async def _execute(self, **kwargs):
            return kwargs

    print_json(DemoTool().to_openai_schema())


if __name__ == "__main__":
    run_menu("tools/base_tool.py", auto_test, manual_test)
