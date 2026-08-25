# agents/summarize_validator_agent.py

from .agent_base import AgentBase


class SummarizeValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(
            name="SummarizeValidatorAgent",
            max_retries=max_retries,
            verbose=verbose
        )

    def execute(self, original_text, summary):
        system_message = (
            "You are a scientific research analysis validator. "
            "Check whether an analysis is accurate, complete, and fully supported "
            "by the original research text."
        )

        user_content = (
            "Evaluate the research analysis against the original text.\n"
            "Check the following criteria:\n"
            "1. It contains Research objective, Methods, Key findings, and Limitations.\n"
            "2. Every claim is supported by the original text.\n"
            "3. Missing information is labeled 'Not stated' rather than invented.\n"
            "4. Important information is not omitted.\n\n"
            "Provide:\n"
            "- A brief validation assessment\n"
            "- Any missing or unsupported information\n"
            "- A rating from 1 to 5, where 5 means excellent quality\n\n"
            f"Original research text:\n{original_text}\n\n"
            f"Research analysis:\n{summary}\n\n"
            "Validation:"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        validation = self.call_llama(messages, max_tokens=512)
        return validation