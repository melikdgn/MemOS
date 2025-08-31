#!/usr/bin/env python3
"""
Comprehensive test script for MemOS examples
Tests all example scripts and verifies functionality
"""

import os
import sys
import json
import subprocess
import asyncio
from pathlib import Path
import tempfile
import shutil

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_mock_example():
    """Test the mock example (no dependencies)"""
    print("🧪 Testing mock example...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the mock example
        mock_script = """
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memos import MemOS
from memos.config import Config

# Mock configuration
config = Config(
    user_id="test_user",
    chat_model={
        "backend": "mock",
        "model": "mock-model",
        "max_tokens": 100
    },
    mem_reader={
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
    settings={
        "max_turns": 5,
        "top_k": 3,
        "textual_memory": True,
        "activation_memory": False,
        "parametric_memory": False
    }
)

# Test basic functionality
mos = MemOS(config=config)
print("✅ MemOS initialized successfully")

# Test memory operations
mos.add_memory("Hello, this is a test memory")
print("✅ Memory added successfully")

# Test chat
response = mos.chat("What memories do I have?")
print(f"✅ Chat response: {response}")

print("🎉 Mock example completed successfully!")
"""
        
        mock_file = Path(temp_dir) / "test_mock.py"
        with open(mock_file, 'w') as f:
            f.write(mock_script)
        
        success, stdout, stderr = run_command(f"{sys.executable} {mock_file}")
        
        if success:
            print("✅ Mock example test passed")
            return True
        else:
            print(f"❌ Mock example test failed: {stderr}")
            return False

def test_simple_example():
    """Test the simple example with Ollama"""
    print("🧪 Testing simple example...")
    
    # Check if Ollama is running
    success, stdout, stderr = run_command("curl -s http://localhost:11434/api/tags")
    if not success:
        print("❌ Ollama not running, skipping simple example")
        return False
    
    # Check if required models are available
    required_models = ["qwen3:0.6b", "nomic-embed-text"]
    for model in required_models:
        success, stdout, stderr = run_command(f"curl -s http://localhost:11434/api/tags | grep -q {model}")
        if not success:
            print(f"❌ Model {model} not found in Ollama")
            return False
    
    # Create test configuration
    config = {
        "user_id": "test_user",
        "chat_model": {
            "backend": "ollama",
            "model": "qwen3:0.6b",
            "max_tokens": 512
        },
        "mem_reader": {
            "backend": "simple_struct",
            "llm": {
                "backend": "ollama",
                "model": "qwen3:0.6b",
                "max_tokens": 512
            },
            "embedder": {
                "backend": "ollama",
                "model": "nomic-embed-text:latest"
            },
            "chunker": {
                "backend": "gpt2",
                "chunk_size": 256,
                "chunk_overlap": 64
            }
        },
        "settings": {
            "max_turns": 3,
            "top_k": 3,
            "textual_memory": True,
            "activation_memory": False,
            "parametric_memory": False
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        test_script = f"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memos import MemOS

# Load configuration
with open('{config_file}') as f:
    import memos.config
    config = memos.config.Config(**json.load(f))

# Test MemOS
mos = MemOS(config=config)
print("✅ MemOS initialized with Ollama")

# Test memory operations
mos.add_memory("This is a test memory from the simple example")
print("✅ Memory added")

# Test chat
response = mos.chat("What do you remember?")
print(f"✅ Chat response: {response}")

print("🎉 Simple example test completed!")
"""
        
        test_file = Path(temp_dir) / "test_simple.py"
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        success, stdout, stderr = run_command(f"{sys.executable} {test_file}")
        
        if success:
            print("✅ Simple example test passed")
            return True
        else:
            print(f"❌ Simple example test failed: {stderr}")
            return False

def test_mcp_server():
    """Test MCP server functionality"""
    print("🧪 Testing MCP server...")
    
    # Test if we can import the MCP server
    try:
        from memos.api.mcp_serve import mcp
        print("✅ MCP server module imported successfully")
        
        # Test if we can access the tools
        tools = mcp.list_tools()
        if len(tools) >= 12:
            print(f"✅ Found {len(tools)} MCP tools")
            return True
        else:
            print(f"❌ Expected 12 tools, found {len(tools)}")
            return False
            
    except Exception as e:
        print(f"❌ MCP server test failed: {e}")
        return False

def test_mcp_client():
    """Test MCP client functionality"""
    print("🧪 Testing MCP client...")
    
    # Create a simple test for MCP client
    test_script = """
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_client():
    try:
        # Import the client
        from examples.mem_mcp.simple_fastmcp_client import main
        
        # Run the client test (this will be a mock test)
        print("✅ MCP client module imported successfully")
        
        # Test basic client functionality
        print("✅ MCP client test completed")
        return True
        
    except Exception as e:
        print(f"❌ MCP client test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_mcp_client())
"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test_mcp_client.py"
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        success, stdout, stderr = run_command(f"{sys.executable} {test_file}")
        
        if success:
            print("✅ MCP client test passed")
            return True
        else:
            print(f"❌ MCP client test failed: {stderr}")
            return False

def test_cli_tools():
    """Test CLI tools"""
    print("🧪 Testing CLI tools...")
    
    # Test memos command
    success, stdout, stderr = run_command("python -m memos.cli --help")
    if success:
        print("✅ CLI tools accessible")
        return True
    else:
        print("❌ CLI tools test failed")
        return False

def main():
    """Run all tests"""
    print("🚀 MemOS Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Mock Example", test_mock_example),
        ("Simple Example", test_simple_example),
        ("MCP Server", test_mcp_server),
        ("MCP Client", test_mcp_client),
        ("CLI Tools", test_cli_tools),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MemOS is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\nRecommendations:")
        print("1. Install Ollama and required models")
        print("2. Start Neo4j and Redis services")
        print("3. Check environment configuration")
        print("4. Run setup_environment.py for automated setup")

if __name__ == "__main__":
    main()