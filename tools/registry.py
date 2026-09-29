"""Safe, governed tool registry for agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from workflow.errors import UnregisteredToolError


@dataclass
class ToolDefinition:
    name: str
    description: str
    requires_approval: bool
    destructive: bool
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        requires_approval: bool = False,
        destructive: bool = False,
        handler: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        """Can be used as a method or a decorator."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_def = ToolDefinition(
                name=name,
                description=description,
                requires_approval=requires_approval,
                destructive=destructive,
                handler=fn,
            )
            self._tools[name] = tool_def
            return fn

        if handler is not None:
            decorator(handler)
            return handler
        return decorator

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise UnregisteredToolError(
                f"Tool '{name}' is not registered. Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a tool strictly through the registry."""
        tool = self.get(name)
        return tool.handler(*args, **kwargs)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())


# Global default registry instance
registry = ToolRegistry()
