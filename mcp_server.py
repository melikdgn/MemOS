import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastmcp import FastMCP

from memos.mem_os.main import MOS


# Load environment variables from .env file
load_dotenv()

from memos.configs.llm import LLMConfigFactory
from memos.configs.mem_os import MOSConfig

# --- LLM Configuration ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()

def create_mos_config() -> MOSConfig:
    """Creates the MOSConfig based on environment variables."""
    if LLM_BACKEND == "ollama":
        print("Using Ollama backend.")
        llm_config = LLMConfigFactory(
            backend="ollama",
            config={
                "model_name_or_path": os.getenv("OLLAMA_MODEL", "llama3"),
                "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            },
        )
    elif LLM_BACKEND == "openrouter":
        print("Using OpenRouter backend.")
        llm_config = LLMConfigFactory(
            backend="openai",  # OpenRouter uses an OpenAI-compatible API
            config={
                "model_name_or_path": os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo"),
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "api_base": "https://openrouter.ai/api/v1",
            },
        )
    else: # Default to OpenAI
        print("Using OpenAI backend.")
        llm_config = LLMConfigFactory(
            backend="openai",
            config={
                "model_name_or_path": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        )

    return MOSConfig(chat_model=llm_config)


# Initialize the Memory Operating System
print("Initializing MemOS with selected backend...")
config = create_mos_config()
mos = MOS(config)
print("MemOS initialized.")

# Create an MCP server instance
mcp = FastMCP(
    "MemOS MCP Server",
    "A server to interact with MemOS memory through the Model Context Protocol.",
)


def format_memory_item(item: Any, cube_id: str) -> Dict[str, Any]:
    """Helper function to format a memory item for the MCP response."""
    return {
        "id": item.id,
        "timestamp": item.metadata.created_at.isoformat() if item.metadata.created_at else datetime.now().isoformat(),
        "content": item.memory,
        "metadata": {
            "cube_id": cube_id,
            "user_id": item.metadata.user_id,
            "source": item.metadata.source,
        },
    }


@mcp.tool()
async def list_memories() -> List[Dict[str, Any]]:
    """
    List all memory entries from all accessible memory cubes.

    Returns:
        A list of memory entries, each with id, timestamp, content, and metadata.
    """
    all_memories = []
    print("Listing memories...")
    # Iterate through all registered memory cubes
    for cube_id, cube in mos.mem_cubes.items():
        if cube.text_mem:
            try:
                # Retrieve all memories from the textual memory component of the cube
                memories = cube.text_mem.get_all()
                for mem in memories:
                    all_memories.append(format_memory_item(mem, cube_id))
            except Exception as e:
                print(f"Error getting memories from cube {cube_id}: {e}")
    print(f"Found {len(all_memories)} memories.")
    return all_memories


@mcp.tool()
async def add_memory(content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Add a new memory entry.

    Args:
        content: The content of the memory to add.
        metadata: An optional dictionary with additional data, e.g., {"user_id": "user1", "cube_id": "cube1"}.

    Returns:
        A dictionary containing the ID of the newly added memory.
    """
    metadata = metadata or {}
    user_id = metadata.get("user_id")
    cube_id = metadata.get("cube_id")

    print(f"Adding memory for user '{user_id}' in cube '{cube_id}'...")

    # The `add` method in MOS returns a list of IDs of the added memories.
    # We are adding a single piece of content, so we expect one ID.
    added_ids = mos.add(memory_content=content, mem_cube_id=cube_id, user_id=user_id)

    if added_ids:
        new_id = added_ids[0]
        print(f"Memory added with ID: {new_id}")
        return {"id": new_id}
    else:
        print("Failed to add memory.")
        return {"error": "Failed to add memory."}


@mcp.tool()
async def delete_memory(id: str) -> Dict[str, Any]:
    """
    Delete a memory entry by its ID.

    Args:
        id: The unique identifier of the memory to delete.

    Returns:
        A dictionary confirming the deletion.
    """
    print(f"Attempting to delete memory with ID: {id}")
    # To delete a memory, we need its cube_id.
    # We must find which cube the memory belongs to.
    target_cube_id = None
    for cube_id, cube in mos.mem_cubes.items():
        if cube.text_mem:
            try:
                # Check if the memory exists in this cube
                mem = cube.text_mem.get(id)
                if mem:
                    target_cube_id = cube_id
                    break
            except Exception:
                # `get` might raise an error if the item is not found
                continue

    if target_cube_id:
        try:
            # MOS delete takes a list of memory IDs
            mos.delete(mem_cube_id=target_cube_id, memory_id=id)
            print(f"Memory {id} deleted from cube {target_cube_id}.")
            return {"status": "success", "deleted_id": id}
        except Exception as e:
            print(f"Error deleting memory {id}: {e}")
            return {"status": "error", "message": str(e)}
    else:
        print(f"Memory with ID {id} not found.")
        return {"status": "error", "message": "Memory not found"}


@mcp.tool()
async def search_memory(query: str) -> List[Dict[str, Any]]:
    """
    Search for memories based on a query.

    Args:
        query: The search query.

    Returns:
        A list of matching memory entries.
    """
    print(f"Searching for: '{query}'")
    # The `search` method returns a rich result object.
    # We need to parse it to get the list of memories.
    search_results = mos.search(query)

    formatted_results = []
    if "text_mem" in search_results:
        for cube_result in search_results["text_mem"]:
            cube_id = cube_result.get("cube_id")
            for mem in cube_result.get("memories", []):
                 formatted_results.append(format_memory_item(mem, cube_id))

    print(f"Found {len(formatted_results)} results for query '{query}'.")
    return formatted_results


async def main():
    """Main function to run the MCP server."""
    # The server can be run in different modes (stdio, http, sse).
    # For this task, we will use the http transport to make it accessible over the network.
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8765"))
    print(f"Starting MCP server on http://{host}:{port}")
    print("Available tools: list_memories, add_memory, delete_memory, search_memory")
    await mcp.run_http_async(host=host, port=port)

if __name__ == "__main__":
    # Ensure you have a .env file with your OPENAI_API_KEY
    # Example .env file:
    # OPENAI_API_KEY="your-openai-api-key"

    # To run this server:
    # 1. Make sure you have installed the dependencies: pip install -r requirements.txt
    # 2. Run the script: python mcp_server.py

    asyncio.run(main())
