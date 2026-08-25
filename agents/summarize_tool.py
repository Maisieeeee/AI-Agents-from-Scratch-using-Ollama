# agents/summarize_agent.py

from .agent_base import AgentBase

class SummarizeTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SummarizeTool", max_retries=max_retries, verbose=verbose)

    def execute(self, text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scientific research assistant. "
                    "Analyze research text accurately and use only information supported by the input. "
                    "If information is missing, write 'Not stated'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Analyze the following scientific research text. "
                    "Return exactly four labeled sections:\n"
                    "1. Research objective\n"
                    "2. Methods\n"
                    "3. Key findings\n"
                    "4. Limitations\n\n"
                    "Do not invent details that are not present in the text.\n\n"
                    f"Research text:\n{text}\n\nResearch analysis:"
                )
            }
        ]
        summary = self.call_llama(messages, max_tokens=500)
        return summary
