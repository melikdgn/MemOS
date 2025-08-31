#!/usr/bin/env python3
"""
Complete MemOS demonstration script
This script provides a guided tour through all MemOS features
"""

import os
import sys
import json
import subprocess
import asyncio
from pathlib import Path
import tempfile
import time

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n📍 Step {step}: {description}")

def check_dependencies():
    """Check if required dependencies are available"""
    print_header("Dependency Check")
    
    dependencies = {
        "Python": sys.version,
        "MemOS": None,
        "Ollama": None,
        "Neo4j": None,
        "Redis": None
    }
    
    # Check MemOS
    try:
        import memos
        dependencies["MemOS"] = memos.__version__ if hasattr(memos, '__version__') else "installed"
        print("✅ MemOS package found")
    except ImportError:
        print("❌ MemOS package not found")
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            dependencies["Ollama"] = "running"
            print("✅ Ollama is running")
        else:
            dependencies["Ollama"] = "not responding"
            print("❌ Ollama not responding")
    except:
        dependencies["Ollama"] = "not running"
        print("❌ Ollama not running")
    
    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        driver.verify_connectivity()
        dependencies["Neo4j"] = "running"
        print("✅ Neo4j is running")
        driver.close()
    except:
        dependencies["Neo4j"] = "not running"
        print("❌ Neo4j not running")
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        dependencies["Redis"] = "running"
        print("✅ Redis is running")
    except:
        dependencies["Redis"] = "not running"
        print("❌ Redis not running")
    
    return dependencies

def run_mock_demo():
    """Run a complete mock demonstration"""
    print_header("Mock Demo - No Dependencies Required")
    
    print_step(1, "Creating mock configuration")
    
    mock_config = {
        "user_id": "demo_user",
        "chat_model": {
            "backend": "mock",
            "model": "mock-model",
            "max_tokens": 100
        },
        "mem_reader": {
            "backend": "simple_struct",
            "llm": {
                "backend": "mock",
                "model": "mock-model",
                "max_tokens": 100
            },
            "embedder": {
                "backend": "mock",
                "model": "mock-embedder"
            },
            "chunker": {
                "backend": "gpt2",
                "chunk_size": 50,
                "chunk_overlap": 10
            }
        },
        "settings": {
            "max_turns": 3,
            "top_k": 2,
            "textual_memory": True,
            "activation_memory": False,
            "parametric_memory": False
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "demo_config.json"
        with open(config_file, 'w') as f:
            json.dump(mock_config, f, indent=2)
        
        print_step(2, "Initializing MemOS with mock backend")
        
        demo_script = f"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memos import MemOS
from memos.config import Config

print("🔄 Loading configuration...")
with open('{config_file}') as f:
    config = Config(**json.load(f))

print("🔄 Initializing MemOS...")
mos = MemOS(config=config)
print("✅ MemOS initialized successfully!")

print("🔄 Adding sample memories...")
memories = [
    "Python is a programming language created by Guido van Rossum",
    "MemOS stands for Memory Operating System",
    "LLMs are Large Language Models like GPT, Claude, and Llama",
    "MCP is the Model Context Protocol for AI applications"
]

for i, memory in enumerate(memories, 1):
    mos.add_memory(memory)
    print(f"  ✅ Added memory {i}: {memory[:50]}...")

print("🔄 Searching memories...")
search_results = mos.search_memories("programming")
print(f"  🔍 Found {len(search_results)} memories about programming:")
for result in search_results:
    print(f"    • {result}")

print("🔄 Chatting with MemOS...")
questions = [
    "What is Python?",
    "What does MemOS stand for?",
    "Tell me about LLMs"
]

for question in questions:
    response = mos.chat(question)
    print(f"  💬 Q: {question}")
    print(f"  🤖 A: {response}")

print("🔄 Getting user info...")
user_info = mos.get_user_info()
print(f"  👤 User: {user_info}")

print("🎉 Mock demo completed successfully!")
"""
        
        demo_file = Path(temp_dir) / "demo.py"
        with open(demo_file, 'w') as f:
            f.write(demo_script)
        
        print("🔄 Running mock demo...")
        os.system(f"{sys.executable} {demo_file}")

def run_mcp_demo():
    """Run MCP server and client demonstration"""
    print_header("MCP Demo - Model Context Protocol")
    
    print_step(1, "Starting MCP server")
    print("   Run this in a separate terminal:")
    print("   python -m memos.api.mcp_serve --transport stdio")
    
    print_step(2, "Testing MCP client")
    print("   Run this in another terminal:")
    print("   python examples/mem_mcp/simple_fastmcp_client.py")
    
    print_step(3, "Available MCP tools:")
    print("   • chat - Chat with memory")
    print("   • create_user - Create new user")
    print("   • create_cube - Create memory cube")
    print("   • register_cube - Register cube")
    print("   • search_memories - Search memories")
    print("   • add_memory - Add new memory")
    print("   • get_memory - Get specific memory")
    print("   • update_memory - Update memory")
    print("   • delete_memory - Delete memory")
    print("   • clear_chat_history - Clear chat")
    print("   • dump_cube - Export cube data")
    print("   • share_cube - Share cube with others")

def create_practical_example():
    """Create a practical usage example"""
    print_header("Practical Example - Research Assistant")
    
    print_step(1, "Scenario Setup")
    print("   You are a research assistant helping with AI/ML topics")
    print("   We'll create a knowledge base for machine learning concepts")
    
    print_step(2, "Memory Organization")
    print("   • Topics: Machine Learning, Deep Learning, NLP")
    print("   • Concepts: Neural Networks, Transformers, LLMs")
    print("   • Facts: Specific algorithms, papers, implementations")
    
    print_step(3, "Usage Workflow")
    print("   1. Add research papers and concepts")
    print("   2. Search for related information")
    print("   3. Chat to get explanations")
    print("   4. Update with new findings")
    
    # Create a practical example script
    example_script = '''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memos import MemOS
from memos.config import Config

# Research assistant configuration
config = Config(
    user_id="research_assistant",
    chat_model={
        "backend": "ollama",
        "model": "qwen3:0.6b",
        "max_tokens": 1024
    },
    mem_reader={
        "backend": "simple_struct",
        "llm": {
            "backend": "ollama",
            "model": "qwen3:0.6b",
            "max_tokens": 1024
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
    settings={
        "max_turns": 10,
        "top_k": 5,
        "textual_memory": True,
        "activation_memory": False,
        "parametric_memory": False
    }
)

def research_assistant_demo():
    """Demonstrate research assistant capabilities"""
    mos = MemOS(config=config)
    
    print("🔬 Research Assistant Demo")
    print("=" * 40)
    
    # Add research content
    research_content = [
        "Transformer architecture was introduced in 'Attention Is All You Need' paper",
        "BERT uses bidirectional encoder representations from transformers",
        "GPT models use autoregressive language modeling",
        "Fine-tuning adapts pre-trained models to specific tasks",
        "Few-shot learning enables models to learn from limited examples",
        "Chain-of-thought prompting improves reasoning capabilities",
        "RAG combines retrieval with generation for better answers",
        "Vector databases store embeddings for semantic search"
    ]
    
    print("📚 Adding research content...")
    for content in research_content:
        mos.add_memory(content)
        print(f"  ✅ Added: {content[:50]}...")
    
    print("\\n🔍 Searching for transformer-related content...")
    results = mos.search_memories("transformer")
    for result in results:
        print(f"  📖 {result}")
    
    print("\\n💬 Asking questions...")
    questions = [
        "What is the transformer architecture?",
        "How does BERT work?",
        "What is RAG?",
        "Explain few-shot learning"
    ]
    
    for question in questions:
        response = mos.chat(question)
        print(f"  ❓ {question}")
        print(f"  💡 {response}\\n")

if __name__ == "__main__":
    research_assistant_demo()
'''
    
    with open("research_assistant_demo.py", "w") as f:
        f.write(example_script)
    
    print("✅ Created research_assistant_demo.py")
    print("   Run: python research_assistant_demo.py")

def show_next_steps():
    """Show recommended next steps"""
    print_header("Next Steps")
    
    print("1. 🏃 Quick Start:")
    print("   python run_mock_example.py")
    
    print("\n2. 🔧 Local Setup:")
    print("   python setup_environment.py")
    
    print("\n3. 🧪 Test Everything:")
    print("   python test_all_examples.py")
    
    print("\n4. 📚 Learn More:")
    print("   • Read GETTING_STARTED.md")
    print("   • Check examples/ directory")
    print("   • Try research_assistant_demo.py")
    
    print("\n5. 🚀 Production Setup:")
    print("   • Install Ollama: https://ollama.ai")
    print("   • Set up Neo4j: https://neo4j.com")
    print("   • Configure Redis: https://redis.io")
    print("   • Set environment variables in .env")

def main():
    """Main demonstration"""
    print("🎯 MemOS Complete Demonstration")
    print("=" * 60)
    
    # Check dependencies
    deps = check_dependencies()
    
    # Run mock demo (always works)
    run_mock_demo()
    
    # Show MCP demo
    run_mcp_demo()
    
    # Create practical example
    create_practical_example()
    
    # Show next steps
    show_next_steps()
    
    print("\n🎉 Demo completed! Check the generated files and follow the next steps.")

if __name__ == "__main__":
    main()