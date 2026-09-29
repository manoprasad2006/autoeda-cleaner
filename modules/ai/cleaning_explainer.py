from __future__ import annotations

from modules.ai.client import GeminiClient
from modules.ai.prompt_templates import build_prompt
from modules.cleaner import CleaningAction

_INSTRUCTION = (
    "Below is one automated data cleaning action, already performed and "
    "logged with a technical reason. Write a short (2-4 sentence) "
    "business-friendly explanation covering: why this issue matters, "
    "what would have gone wrong if left unfixed, and any real trade-off "
    "of the method chosen. Do not just restate the technical reason "
    "verbatim -- add genuine business/analytical context."
)


def build_cleaning_explanation_prompt(action: CleaningAction) -> str:
    context = (
        f"Column: {action.column}\n"
        f"Issue: {action.issue}\n"
        f"Method used: {action.method}\n"
        f"Technical reason: {action.reason}\n"
        f"Before: {action.before_summary}\n"
        f"After: {action.after_summary}\n"
        f"Rows affected: {action.rows_affected}"
    )
    return build_prompt(_INSTRUCTION, context)


def explain_cleaning_action(client: GeminiClient, action: CleaningAction) -> str:
    """Get an AI-generated explanation for one cleaning action.

    On failure (no API key, network issue, rate limit exhausted), falls
    back to the action's own technical reason rather than raising -- a
    missing AI explanation should never break the Cleaning page, since
    the factual reason is already genuinely useful on its own."""
    prompt = build_cleaning_explanation_prompt(action)
    response = client.generate(prompt)

    if response.success:
        return response.text

    return action.reason


def action_key(action: CleaningAction, index: int) -> str:
    """Build a unique dictionary key for one action within a log.

    A log can have multiple actions on the same column (e.g. both a
    whitespace fix and a casing fix on 'country'), so column name alone
    isn't a safe key. Exposed as its own function -- rather than being
    private to explain_all_actions() -- so callers (like the Cleaning
    page) can look up an explanation using the exact same key logic,
    instead of re-deriving their own copy of this format string."""
    return f"{action.column}::{action.issue}::{index}"


def explain_all_actions(
    client: GeminiClient, actions: list[CleaningAction]
) -> dict[str, str]:
    """Explain every action in a cleaning log, keyed by action_key()."""
    explanations: dict[str, str] = {}

    for i, action in enumerate(actions):
        explanations[action_key(action, i)] = explain_cleaning_action(client, action)

    return explanations
