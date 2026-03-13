import textwrap
from agentdojo.logging import OutputLogger


class FullOutputLogger(OutputLogger):
    """
    A non-truncating OutputLogger that prevents AgentDojo from hiding context payload strings
    using textwrap.shorten [...]. This executes cleanly without modifying AgentDojo source code.
    """

    def log(self, *args, **kwargs) -> None:
        original_shorten = textwrap.shorten
        textwrap.shorten = lambda text, *a, **kw: text
        try:
            super().log(*args, **kwargs)
        finally:
            textwrap.shorten = original_shorten
