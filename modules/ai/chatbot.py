from __future__ import annotations

from dataclasses import dataclass, field

from modules.ai.client import GeminiClient
from modules.ai.executive_summary import _plural
from modules.cleaner import CleaningLog
from modules.eda import EDAResult
from modules.ml_readiness import MLReadinessResult
from modules.profiler import DatasetProfile
from modules.quality import QualityScore

_SYSTEM_CONTEXT_TEMPLATE = """You are a data analyst assistant answering questions about a specific dataset. Only answer based on the facts below -- if something isn't covered by these facts, say so honestly rather than guessing.

DATASET FACTS:
- Shape: {n_rows} {row_word}, {n_columns} {column_word}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Date columns: {date_cols}
- Data quality score: {quality_score}% (completeness {completeness}%, duplicates {duplicate_score}%, consistency {consistency}%, validity {validity}%)
- Cleaning actions taken: {cleaning_summary}
- Correlations found: {correlations}
- ML readiness score: {readiness_score}% (task: {inferred_task})
"""


def build_system_context(
    profile: DatasetProfile,
    quality: QualityScore,
    log: CleaningLog,
    eda: EDAResult,
    readiness: MLReadinessResult | None,
) -> str:
    cleaning_summary = (
        ", ".join(sorted({a.issue for a in log.actions}))
        if log.actions
        else "none applied"
    )
    correlations = (
        "; ".join(
            f"{p.column_a}/{p.column_b} ({p.correlation})"
            for p in eda.correlation_pairs[:5]
        )
        if eda.correlation_pairs
        else "none found"
    )

    return _SYSTEM_CONTEXT_TEMPLATE.format(
        n_rows=profile.n_rows,
        n_columns=profile.n_columns,
        row_word=_plural(profile.n_rows, "row"),
        column_word=_plural(profile.n_columns, "column"),
        numeric_cols=", ".join(profile.numerical_columns) or "none",
        categorical_cols=", ".join(profile.categorical_columns) or "none",
        date_cols=", ".join(profile.date_columns) or "none",
        quality_score=quality.overall_score,
        completeness=quality.completeness_score,
        duplicate_score=quality.duplicate_score,
        consistency=quality.consistency_score,
        validity=quality.validity_score,
        cleaning_summary=cleaning_summary,
        correlations=correlations,
        readiness_score=readiness.readiness_score if readiness else "(not assessed)",
        inferred_task=readiness.inferred_task if readiness else "(not assessed)",
    )


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ChatSession:
    """Holds conversation history and the dataset context it's grounded in."""

    system_context: str
    messages: list[ChatMessage] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=content))

    def build_prompt(self) -> str:
        """Combine system context with the full conversation so far into
        one prompt string. Simple text-based history rather than the
        SDK's native multi-turn chat objects, matching this project's
        existing GeminiClient.generate(prompt) interface."""
        history = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in self.messages[-12:]
        )
        return f"{self.system_context}\n\nConversation so far:\n{history}\n\nAssistant:"


def ask_chatbot(client: GeminiClient, session: ChatSession, question: str) -> str:
    """Add the user's question to the session, get a response, add that
    response to the session too, and return it. On AI failure, returns
    an honest error message rather than crashing the chat UI -- there's
    no meaningful structural fallback for open-ended Q&A."""
    if len(question) > 2000:
        return "Please keep your question under 2,000 characters."
    session.add_user_message(question)
    prompt = session.build_prompt()
    response = client.generate(prompt)

    if response.success:
        answer = response.text
    else:
        answer = (
            "Sorry, I couldn't reach the AI service right now "
            f"({response.error}). Please try again in a moment."
        )

    session.add_assistant_message(answer)
    return answer
