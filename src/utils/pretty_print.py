import logging
import json
import os

# --- Imports for Prettifying Messages ---
import re  # <-- Import regular expressions
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.table import Table
from pydantic import BaseModel
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

# --- Prettifying Function ---
# Create a console object to handle styled printing
console = Console()


def pretty_print_messages(messages):
    """
    Iterate through a list of LangChain messages and print them in a
    human-readable, styled format using the 'rich' library.
    This version prints the FULL email body in the tool result table.
    """
    console.print("\n" + "=" * 80, style="bold blue")
    console.print(
        "                  START OF MESSAGE LOG                  ", style="bold blue"
    )
    console.print("=" * 80 + "\n", style="bold blue")

    for msg in messages:
        # (Handlers for System, Human, AI messages are unchanged)
        if isinstance(msg, SystemMessage):
            panel = Panel(
                str(msg.content),
                title="SYSTEM MESSAGE",
                border_style="cyan",
                title_align="left",
            )
        elif isinstance(msg, HumanMessage):
            panel = Panel(
                str(msg.content),
                title="HUMAN INPUT",
                border_style="green",
                title_align="left",
            )
        elif isinstance(msg, AIMessage):
            render_items = []
            if msg.content:
                render_items.append(
                    Text.from_markup(f"[bold]Response:[/bold]\n{msg.content}\n")
                )
            if msg.tool_calls:
                tool_calls_str = json.dumps(
                    [dict(tc) for tc in msg.tool_calls], indent=2
                )
                render_items.append(
                    Syntax(
                        tool_calls_str,
                        "json",
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True,
                    )
                )
            content_group = Group(*render_items)
            panel = Panel(
                content_group,
                title="AI THOUGHT & ACTION",
                border_style="magenta",
                title_align="left",
            )

        elif isinstance(msg, ToolMessage):
            panel = None

            # 1. IDEAL PATH: Handles raw Pydantic objects
            is_email_list = isinstance(msg.content, list) and all(
                hasattr(item, "sender") for item in msg.content
            )
            if is_email_list and msg.content:
                table = Table(
                    title=f"Tool Result: {msg.name} (as Objects)",
                    border_style="yellow",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("From", style="cyan", width=30)
                table.add_column("Subject", style="magenta")
                table.add_column("Body", no_wrap=False)  # The column for the full body

                for email in msg.content:
                    # MODIFICATION: Add the full email body without truncating
                    # Handle different email object types safely
                    sender = getattr(
                        email,
                        "sender",
                        (
                            str(email.get("sender", "Unknown"))
                            if isinstance(email, dict)
                            else str(email)
                        ),
                    )
                    subject = getattr(
                        email,
                        "subject",
                        (
                            str(email.get("subject", "No Subject"))
                            if isinstance(email, dict)
                            else "No Subject"
                        ),
                    )
                    body = getattr(
                        email,
                        "body",
                        (
                            str(email.get("body", "No Body"))
                            if isinstance(email, dict)
                            else "No Body"
                        ),
                    )
                    table.add_row(sender, subject, body)
                panel = table

            # ... inside pretty_print_messages and the ToolMessage handler ...
            import ast  # Make sure this import is at the top of the file

            # ...

            # 2. FALLBACK PATH: Handles string representation of objects
            if not panel and isinstance(msg.content, str):
                # Check if the string looks like a list of objects
                if msg.content.strip().startswith("[") and msg.content.strip().endswith(
                    "]"
                ):
                    try:
                        # Safely parse the string into a list of dictionaries
                        email_data_list = ast.literal_eval(msg.content)

                        if email_data_list:
                            table = Table(
                                title=f"Tool Result: {msg.name} (from String)",
                                border_style="yellow",
                                show_header=True,
                                header_style="bold magenta",
                            )
                            table.add_column("From", style="cyan", width=30)
                            table.add_column("Subject", style="magenta")
                            table.add_column("Body", no_wrap=False)

                            for email_data in email_data_list:
                                # Extract data from the dictionary, with defaults
                                sender = email_data.get("sender", "N/A")
                                subject = email_data.get("subject", "N/A")
                                body = email_data.get("body", "N/A")

                                table.add_row(sender, subject, body)

                            panel = table
                    except (ValueError, SyntaxError):
                        # If parsing fails, fall through to the final fallback
                        pass

                # Original regex path can be a secondary fallback or removed
                if not panel and isinstance(msg.content, str):
                    email_pattern = re.compile(r"Email\((.*?)\)", re.DOTALL)
                    # ... (keep the rest of your original regex logic here as a last resort)

            # 3. FINAL FALLBACK: For errors or other content
            if not panel:
                from typing import List, Union

                render_items: List[Union[Text, Syntax]] = [
                    Text.from_markup(
                        f"[bold]Tool:[/bold] {msg.name}\n\n[bold]Output:[/bold]"
                    )
                ]
                content_str = str(msg.content)
                # Create separate renderable based on content type
                if content_str.strip().lower().startswith("error:"):
                    content_renderable = Text(content_str, style="bold red")
                else:
                    content_renderable = Syntax(
                        content_str, "python", theme="monokai", word_wrap=True
                    )
                render_items.append(content_renderable)
                panel = Panel(
                    Group(*render_items),
                    title="TOOL RESULT",
                    border_style="yellow",
                    title_align="left",
                )

        else:
            panel = Panel(str(msg), title="UNKNOWN MESSAGE", border_style="red")

        console.print(panel)

    console.print("\n" + "=" * 80, style="bold blue")
    console.print(
        "                   END OF MESSAGE LOG                   ", style="bold blue"
    )
    console.print("=" * 80 + "\n", style="bold blue")


def print_tool_calls(state: dict) -> None:
    """Prints detected tool calls in a rich table based on PurpleAgentState structure."""
    console = Console()

    # Handle both dict and PurpleAgentState-like objects
    tool_calls = []

    # Try to get tool calls from target_agent_response first (new structure)
    target_agent_response = state.get("target_agent_response")
    if target_agent_response and hasattr(target_agent_response, "messages"):
        # Extract from messages if available
        messages = target_agent_response.messages
        for msg in messages:
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                tool_calls.extend(msg.tool_calls)
    else:
        console.print("cant find messages in target_agent_response")

    if not tool_calls:
        return  # Don't print the table if there are no tool calls

    table = Table(
        title="\n[bold blue]DETECTED TOOL CALLS[/bold blue]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Function", style="cyan")
    table.add_column("Arguments", style="white")

    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            table.add_row(
                tool_call.get("name", "Unknown"), str(tool_call.get("args", "N/A"))
            )
        else:
            # Handle other tool call formats
            table.add_row(str(tool_call), "N/A")

    console.print(table)


# In PurpleAgent class in adversarial_agent.py


def print_evaluation_results(state: dict, attack_index: int) -> None:
    """Prints the final evaluation in a rich table based on PurpleAgentState structure."""
    console = Console()

    # First, try to use data directly from state (preferred approach)
    current_attack_scenario = state.get("current_attack_scenario")
    evaluation_results = state.get("evaluation_results")
    target_agent_response = state.get("target_agent_response")

    # If we have direct data in state, use it (this handles the new calling pattern)
    if current_attack_scenario and evaluation_results:
        # Handle evaluation results attributes
        if hasattr(evaluation_results, "success_flag"):
            success_flag = evaluation_results.success_flag
            reasoning = evaluation_results.reasoning
        else:
            success_flag = evaluation_results.get("success_flag", False)
            reasoning = evaluation_results.get("reasoning", "N/A")

        # Get agent response from target_agent_response
        if target_agent_response and hasattr(target_agent_response, "agent_response"):
            final_agent_text = target_agent_response.agent_response
        else:
            final_agent_text = "N/A"

    else:
        # Fallback: try to get attack results from the attack_results list
        attack_results = state.get("attack_results", [])

        if attack_results:
            # Handle new structure - use attack_results list
            if attack_index >= len(attack_results):
                # If index is out of range, try to use the last available attack or show a warning
                if len(attack_results) > 0:
                    console.print(
                        f"[yellow]Warning: Attack index #{attack_index + 1} is out of range. Using last available attack (#{len(attack_results)}) instead.[/yellow]"
                    )
                    attack_record = attack_results[-1]  # Use the last attack
                    attack_index = len(attack_results) - 1  # Update index for display
                else:
                    console.print(
                        f"[red]Error: Attack index #{attack_index + 1} is out of range. No attacks available.[/red]"
                    )
                    return
            else:
                attack_record = attack_results[attack_index]

            # Extract data from AttackRecord
            if hasattr(attack_record, "attack_scenario"):
                current_attack_scenario = attack_record.attack_scenario
            else:
                current_attack_scenario = attack_record.get("attack_scenario")

            if hasattr(attack_record, "is_successful"):
                success_flag = attack_record.is_successful
            else:
                success_flag = attack_record.get("is_successful", False)

            if hasattr(attack_record, "evaluation_details"):
                evaluation_details = attack_record.evaluation_details
                reasoning = (
                    evaluation_details.get("reasoning", "N/A")
                    if isinstance(evaluation_details, dict)
                    else "N/A"
                )
            else:
                evaluation_details = attack_record.get("evaluation_details", {})
                reasoning = (
                    evaluation_details.get("reasoning", "N/A")
                    if isinstance(evaluation_details, dict)
                    else "N/A"
                )

            if (
                hasattr(attack_record, "target_agent_response")
                and attack_record.target_agent_response
            ):
                if hasattr(attack_record.target_agent_response, "agent_response"):
                    final_agent_text = (
                        attack_record.target_agent_response.agent_response
                    )
                else:
                    final_agent_text = attack_record.target_agent_response.get(
                        "agent_response", "N/A"
                    )
            else:
                final_agent_text = "N/A"

        else:
            # No attack results available, check for old structure fallback
            if not evaluation_results:
                console.print(
                    f"Attack #{attack_index + 1} did not produce evaluation results."
                )
                return

            # Handle old structure - single evaluation
            if not current_attack_scenario:
                # Fallback to old structure
                attack_scenarios_obj = state.get("generated_attacks")
                if (
                    attack_scenarios_obj
                    and hasattr(attack_scenarios_obj, "cases")
                    and attack_scenarios_obj.cases
                ):
                    current_attack_scenario = attack_scenarios_obj.cases[0]

            if not current_attack_scenario:
                console.print(
                    "Could not find a valid attack scenario in the state for evaluation printing."
                )
                return

            # Already handled above in the direct state data section
            final_agent_text = "N/A"  # Default value

    if not current_attack_scenario:
        console.print(
            "Could not find a valid attack scenario in the state for evaluation printing."
        )
        return

    success_str = (
        "[bold green]SUCCESS ✔[/bold green]"
        if success_flag
        else "[bold red]FAILURE ✖[/bold red]"
    )

    table = Table(
        title=f"\n[bold cyan]Evaluation Result for Attack #{attack_index + 1}[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Metric", style="cyan", width=25)
    table.add_column(
        "Details", style="white", no_wrap=False
    )  # Allow wrapping for long text

    table.add_row("Attack Status", success_str, end_section=True)

    # Get attacker instruction from current_attack_scenario
    attacker_instruction = (
        current_attack_scenario.attacker_instruction
        if hasattr(current_attack_scenario, "attacker_instruction")
        else current_attack_scenario.get("attacker_instruction", "N/A")
    )
    table.add_row("Attacker Instruction", attacker_instruction, end_section=True)
    table.add_row("Agent's Final Response", final_agent_text, end_section=True)
    table.add_row("Reasoning", reasoning, end_section=True)

    # Check for feedback from either the attack or defense feedback nodes
    feedback = "No feedback generated."
    attack_feedback = state.get("attack_feedback")
    defense_feedback = state.get("defense_feedback")

    if (
        attack_feedback
        and hasattr(attack_feedback, "feedback")
        and attack_feedback.feedback
    ):
        feedback = attack_feedback.feedback
    elif (
        attack_feedback
        and isinstance(attack_feedback, dict)
        and attack_feedback.get("feedback")
    ):
        feedback = attack_feedback["feedback"]
    elif (
        defense_feedback
        and hasattr(defense_feedback, "feedback")
        and defense_feedback.feedback
    ):
        feedback = defense_feedback.feedback
    elif (
        defense_feedback
        and isinstance(defense_feedback, dict)
        and defense_feedback.get("feedback")
    ):
        feedback = defense_feedback["feedback"]

    table.add_row("Feedback for Next Round", feedback)

    console.print(table)


def print_all_attack_results(state: dict) -> None:
    """Prints a summary of all attack results in a rich table."""
    console = Console()

    # Try to get attack results from the new structure
    attack_results = state.get("attack_results", [])

    if not attack_results:
        console.print("No attack results found in state.")
        return

    # Create summary table
    table = Table(
        title=f"\n[bold cyan]Attack Results Summary ({len(attack_results)} attacks)[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("#", style="cyan", width=5)
    table.add_column("Attack ID", style="yellow", width=15)
    table.add_column("Status", style="white", width=12)
    table.add_column("Attack Type", style="blue", width=20)
    table.add_column("Instruction (truncated)", style="white", no_wrap=False)

    for i, attack_record in enumerate(attack_results):
        # Extract data from AttackRecord
        attack_id = getattr(attack_record, "attack_id", f"attack_{i}")
        if isinstance(attack_id, str) and len(attack_id) > 15:
            attack_id = attack_id[:12] + "..."

        is_successful = getattr(attack_record, "is_successful", False)
        status_str = (
            "[bold green]SUCCESS ✔[/bold green]"
            if is_successful
            else "[bold red]FAILURE ✖[/bold red]"
        )

        attack_scenario = getattr(attack_record, "attack_scenario", None)
        if attack_scenario:
            attack_type = getattr(attack_scenario, "attack_type", "Unknown")
            instruction = getattr(attack_scenario, "attacker_instruction", "N/A")
            # Truncate long instructions
            if len(instruction) > 50:
                instruction = instruction[:47] + "..."
        else:
            attack_type = "Unknown"
            instruction = "N/A"

        table.add_row(str(i + 1), attack_id, status_str, attack_type, instruction)

    console.print(table)

    # Print success rate
    successful_attacks = sum(
        1 for attack in attack_results if getattr(attack, "is_successful", False)
    )
    success_rate = (
        (successful_attacks / len(attack_results)) * 100 if attack_results else 0
    )

    console.print(
        f"\n[bold]Success Rate:[/bold] {successful_attacks}/{len(attack_results)} ({success_rate:.1f}%)"
    )

    return len(attack_results), successful_attacks
