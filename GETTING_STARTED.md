# MemOS Getting Started Guide

## Overview
MemOS (Memory Operating System) is a memory system for LLMs that provides modular memory containers called MemCubes. It supports multiple memory types (textual, activation, parametric, KV cache) and offers significant performance improvements over standard LLM implementations.

## Performance
- **38.98% improvement** over OpenAI baseline
- **159% improvement** in temporal reasoning
- **Modular architecture** with pluggable backends

## Installation

### Quick Install via pip
```bash
pip install MemoryOS
```

### Install with Optional Dependencies
```bash
# Basic installation
pip install MemoryOS

# With tree memory support (Neo4j)
pip install MemoryOS[tree-mem]

# With memory scheduler (Redis)
pip install MemoryOS[mem-scheduler]

# With document processing
pip install MemoryOS[mem-reader]

# All features
pip install MemoryOS[all]
```

### Development Installation
```bash
git clone https://github.com/MemTensor/MemOS.git
cd MemOS
pip install -e .
```

## Prerequisites

### Required External Services
1. **Ollama** (for local models)
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull required models
   ollama pull qwen3:0.6b
   ollama pull nomic-embed-text
   ```

2. **Neo4j** (for tree memory, optional)
   ```bash
   # Using Docker
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/your-password \
     neo4j:latest
   ```

3. **Redis** (for memory scheduler, optional)
   ```bash
   # Using Docker
   docker run -d \
     --name redis \
     -p 6379:6379 \
     redis:latest
   ```

## Configuration

### 1. Environment Setup
Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your actual values:
```bash
# Required: OpenAI API key or Ollama configuration
OPENAI_API_KEY=your_openai_api_key_here
# OR
OLLAMA_BASE_URL=http://localhost:11434

# Optional: External services
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 2. Configuration File
Create `config.json`:
```json
{
  "user_id": "your_user_id",
  "chat_model": {
    "backend": "huggingface",
    "model": "Qwen/Qwen3-1.7B",
    "max_tokens": 4096
  },
  "mem_reader": {
    "backend": "simple_struct",
    "llm": {
      "backend": "ollama",
      "model": "qwen3:0.6b",
      "max_tokens": 8192
    },
    "embedder": {
      "backend": "ollama",
      "model": "nomic-embed-text:latest"
    },
    "chunker": {
      "backend": "gpt2",
      "chunk_size": 512,
      "chunk_overlap": 128
    }
  },
  "settings": {
    "max_turns": 20,
    "top_k": 5,
    "textual_memory": true,
    "activation_memory": false,
    "parametric_memory": false
  }
}
```

## Quick Start Examples

### 1. Basic Usage
```python
from memos import MemOS

# Initialize with default configuration
mos = MemOS()

# Create a memory cube
cube = mos.create_cube("my_cube")

# Add memory
cube.add_memory("Python is a programming language created by Guido van Rossum.")

# Chat with memory
response = cube.chat("Who created Python?")
print(response)
```

### 2. Using Configuration File
```python
from memos import MemOS
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize with custom configuration
mos = MemOS(config=config)

# Create user and cube
user = mos.create_user("alice")
cube = mos.create_cube("alice_cube", user_id=user.id)

# Add structured memory
cube.add_memory(
    content="Machine learning is a subset of AI that enables systems to learn from data.",
    metadata={"topic": "AI", "category": "technology"}
)

# Search memories
results = cube.search_memories("artificial intelligence")
for memory in results:
    print(f"Content: {memory.content}")
    print(f"Relevance: {memory.relevance_score}")
```

### 3. MCP Server Usage
Start the MCP server:
```bash
# Via CLI
memos mcp_serve --transport stdio

# Via Python
python -m memos.api.mcp_serve --transport stdio
```

Use the MCP client:
```python
from examples.mem_mcp.simple_fastmcp_client import MemOSClient

async def main():
    client = MemOSClient()
    
    # Create user
    user = await client.create_user("test_user")
    
    # Create cube
    cube = await client.create_cube("test_cube", user_id=user["id"])
    
    # Add memory
    await client.add_memory(
        cube_id=cube["id"],
        content="The capital of France is Paris."
    )
    
    # Chat with context
    response = await client.chat(
        cube_id=cube["id"],
        message="What is the capital of France?"
    )
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Memory Types

### 1. Textual Memory
- **Description**: Stores text-based information
- **Backend**: Neo4j graph database
- **Use Case**: Facts, conversations, documents

### 2. Activation Memory
- **Description**: Stores model activations for faster inference
- **Backend**: Local storage
- **Use Case**: Caching, performance optimization

### 3. Parametric Memory
- **Description**: Stores model parameters (LoRA adapters)
- **Backend**: Local storage
- **Use Case**: Model fine-tuning, personalization

### 4. KV Cache Memory
- **Description**: Key-value storage for quick lookups
- **Backend**: Redis
- **Use Case**: Session management, temporary storage

## CLI Usage

### Available Commands
```bash
# Download examples
memos download_examples

# Export OpenAPI schema
memos export_openapi

# Run MCP server
memos mcp_serve --transport stdio
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'memos'**
   ```bash
   pip install MemoryOS
   ```

2. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Pull required models
   ollama pull qwen3:0.6b
   ollama pull nomic-embed-text
   ```

3. **Neo4j Connection Error**
   ```bash
   # Check Neo4j status
   docker ps | grep neo4j
   
   # Verify connection
   cypher-shell -u neo4j -p your_password
   ```

4. **Memory Not Found Error**
   - Ensure the cube is properly registered
   - Check user permissions
   - Verify memory ID exists

## Performance Tips

1. **Use Local Models**: Ollama provides better performance for local development
2. **Enable Caching**: Use activation memory for frequently accessed data
3. **Optimize Chunking**: Adjust chunk_size and chunk_overlap for your use case
4. **Connection Pooling**: Configure connection pools for external services

## Next Steps

1. **Explore Examples**: Check the `examples/` directory for more usage patterns
2. **Read Documentation**: Visit https://memos-docs.openmem.net/
3. **Join Community**: https://github.com/MemTensor/MemOS/discussions
4. **Contribute**: Submit issues and pull requests

## Support

- **Documentation**: https://memos-docs.openmem.net/
- **Issues**: https://github.com/MemTensor/MemOS/issues
- **Discussions**: https://github.com/MemTensor/MemOS/discussions