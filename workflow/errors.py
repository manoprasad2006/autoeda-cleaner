"""Workflow error definitions."""
from __future__ import annotations


class WorkflowError(Exception):
    """Base error for workflow execution."""


class InvalidStageTransitionError(WorkflowError):
    """Raised when an illegal stage transition is requested."""


class ApprovalRequiredError(WorkflowError):
    """Raised when attempting an unapproved consequential action."""


class ProposalNotFoundError(WorkflowError):
    """Raised when an approval references a non-existent proposal."""


class UnregisteredToolError(WorkflowError):
    """Raised when an agent attempts to invoke an unregistered tool."""


class MaxStepsExceededError(WorkflowError):
    """Raised when supervisor loop exceeds the configured step ceiling."""


class RollbackError(WorkflowError):
    """Raised when rollback cannot be performed (e.g., at root version)."""
