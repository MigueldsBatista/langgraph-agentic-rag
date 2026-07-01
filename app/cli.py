from typing import Any

class CommandRegistry:
    """Manages interactive CLI commands separately from graph logic."""
    
    def __init__(self):
        self._commands = {}

    def register(self, name: str, description: str):
        def decorator(func):
            self._commands[name] = {"func": func, "description": description}
            return func
        return decorator

    def handle(self, user_input: str, context: dict[str, Any]) -> bool:
        """Processes the command if recognized. Returns True if handled, False otherwise."""
        tokens = user_input.strip().split()
        if not tokens:
            return False
        
        cmd = tokens[0].lower()
        if cmd in self._commands:
            args = tokens[1:]
            self._commands[cmd]["func"](args, context)
            return True
        return False

    def get_help(self) -> str:
        help_lines = ["\n--- CLI Commands ---"]
        for name, info in sorted(self._commands.items()):
            help_lines.append(f"  {name:<12} - {info['description']}")
        help_lines.append("")
        return "\n".join(help_lines)


registry = CommandRegistry()


@registry.register("/help", "Show all available commands.")
def handle_help(args: list[str], context: dict[str, Any]):
    print(registry.get_help())


@registry.register("/exit", "Exit the conversation session.")
@registry.register("/quit", "Exit the conversation session.")
def handle_exit(args: list[str], context: dict[str, Any]):
    context["should_exit"] = True
    print("Goodbye!")


@registry.register("/history", "List conversation checkpoints from newest to oldest.")
def handle_history(args: list[str], context: dict[str, Any]):
    graph = context["graph"]
    session_id = context["session_id"]
    active_checkpoint = context["active_checkpoint"]
    
    config = {"configurable": {"thread_id": session_id}}
    history = graph.get_state_history(config)
    
    print("\n--- Checkpoint History (Newest to Oldest) ---")
    for state in history:
        cid = state.config["configurable"]["checkpoint_id"]
        last_msg = (
            state.values["messages"][-1].content[:50]
            if state.values.get("messages")
            else "No messages"
        )
        active_indicator = " -> (ACTIVE)" if cid == active_checkpoint else ""
        print(f"ID: {cid}{active_indicator}")
        print(f"   Last output: {last_msg}...")
        print(f"   Next node scheduled: {state.next or 'None (Ended)'}\n")


@registry.register("/travel", "Time-travel to a checkpoint: /travel <checkpoint_id>")
def handle_travel(args: list[str], context: dict[str, Any]):
    if not args:
        print("❌ Error: Please specify a checkpoint ID. Example: /travel <id>\n")
        return
    
    target_id = args[0].strip()
    graph = context["graph"]
    session_id = context["session_id"]
    
    config = {"configurable": {"thread_id": session_id}}
    history = list(graph.get_state_history(config))
    valid_ids = [state.config["configurable"]["checkpoint_id"] for state in history]
    
    if target_id in valid_ids:
        context["active_checkpoint"] = target_id
        print(f"🚀 Teleported back to checkpoint: {target_id}\n")
    else:
        print(f"❌ Invalid Checkpoint ID. Valid checkpoints: {valid_ids}\n")
