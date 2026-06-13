from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("Wake Gemini")

@mcp.tool()
def wakeup(priority: int = 1) -> str:
    """
    Dummy tool that serves as a signaling mechanism.
    When an external script needs Gemini to wake up, it cannot call this tool directly 
    (since MCP tools are called BY Gemini, not the other way around). 
    However, if an external script interacts with a shared state (e.g., touches a 
    specific file), a background task running in Gemini's environment can detect that 
    change and trigger a notification, effectively 'waking' Gemini.
    """
    return f"Wakeup signal received with priority {priority}"

if __name__ == "__main__":
    mcp.run()
