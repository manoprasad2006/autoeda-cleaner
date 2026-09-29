"""Base agent interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from agents.schemas import AgentResult
from workflow.state import WorkflowState


class BaseAgent(ABC):
    name: str = "BaseAgent"
    description: str = "Abstract agent base class"

    @abstractmethod
    def run(self, state: WorkflowState) -> AgentResult:
        """Run the agent on current workflow state and return structured result.
        Must not directly mutate state.active_dataset unless explicitly approved executor.
        Must use registered tools."""
        raise NotImplementedError
