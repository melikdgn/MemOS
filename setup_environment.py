#!/usr/bin/env python3
"""
MemOS Environment Setup Script
Automatically configures the environment for MemOS based on available services
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def check_command(cmd):
    """Check if a command is available in the system"""
    try:
        subprocess.run([cmd, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_service(url, service_name):
    """Check if a service is running"""
    import requests
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def setup_ollama():
    """Setup Ollama with required models"""
    print("🔧 Setting up Ollama...")
    
    if not check_command('ollama'):
        print("❌ Ollama not found. Please install Ollama first:")
        print("   curl -fsSL https://ollama.com/install.sh | sh")
        return False
    
    models = ['qwen3:0.6b', 'nomic-embed-text']
    for model in models:
        print(f"📦 Pulling {model}...")
        try:
            subprocess.run(['ollama', 'pull', model], check=True)
            print(f"✅ {model} pulled successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to pull {model}")
            return False
    
    return True

def setup_neo4j():
    """Setup Neo4j database"""
    print("🔧 Setting up Neo4j...")
    
    if check_service('http://localhost:7474', 'Neo4j'):
        print("✅ Neo4j is already running")
        return True
    
    if check_command('docker'):
        print("📦 Starting Neo4j with Docker...")
        try:
            subprocess.run([
                'docker', 'run', '-d',
                '--name', 'memos-neo4j',
                '-p', '7474:7474',
                '-p', '7687:7687',
                '-e', 'NEO4J_AUTH=neo4j/memospassword',
                'neo4j:latest'
            ], check=True)
            print("✅ Neo4j started successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to start Neo4j with Docker")
    
    print("❌ Neo4j not found. Please install Neo4j or use Docker:")
    print("   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your-password neo4j:latest")
    return False

def setup_redis():
    """Setup Redis for memory scheduler"""
    print("🔧 Setting up Redis...")
    
    if check_service('http://localhost:6379', 'Redis'):
        print("✅ Redis is already running")
        return True
    
    if check_command('docker'):
        print("📦 Starting Redis with Docker...")
        try:
            subprocess.run([
                'docker', 'run', '-d',
                '--name', 'memos-redis',
                '-p', '6379:6379',
                'redis:latest'
            ], check=True)
            print("✅ Redis started successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to start Redis with Docker")
    
    print("❌ Redis not found. Please install Redis or use Docker:")
    print("   docker run -d --name redis -p 6379:6379 redis:latest")
    return False

def create_config():
    """Create configuration file"""
    print("🔧 Creating configuration file...")
    
    config = {
        "user_id": "test_user",
        "chat_model": {
            "backend": "ollama",
            "model": "qwen3:0.6b",
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
            "textual_memory": True,
            "activation_memory": False,
            "parametric_memory": False
        }
    }
    
    config_path = Path("config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_path}")

def create_env_file():
    """Create .env file from example"""
    print("🔧 Creating .env file...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        return
    
    if env_example.exists():
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ .env file created from .env.example")
    else:
        # Create basic .env file
        env_content = """# MemOS Environment Configuration

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Neo4j Configuration (for tree memory)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=memospassword

# Redis Configuration (for memory scheduler)
REDIS_URL=redis://localhost:6379

# Memory Configuration
MOS_TEXT_MEM_TYPE=textual
MOS_TREE_MEM_TYPE=tree
MOS_ACTIVATION_MEM_TYPE=activation
MOS_PARAMETRIC_MEM_TYPE=parametric
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Basic .env file created")

def test_installation():
    """Test if MemOS is properly installed"""
    print("🔧 Testing MemOS installation...")
    
    try:
        import memos
        print("✅ MemOS package imported successfully")
        
        # Test basic functionality
        from memos import MemOS
        mos = MemOS()
        print("✅ MemOS initialized successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import MemOS: {e}")
        return False
    except Exception as e:
        print(f"❌ Error initializing MemOS: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 MemOS Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install MemOS if not already installed
    try:
        import memos
        print("✅ MemOS already installed")
    except ImportError:
        print("📦 Installing MemOS...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'MemoryOS'], check=True)
    
    # Setup services
    services = []
    
    if input("Setup Ollama? (y/n): ").lower() == 'y':
        services.append(setup_ollama())
    
    if input("Setup Neo4j? (y/n): ").lower() == 'y':
        services.append(setup_neo4j())
    
    if input("Setup Redis? (y/n): ").lower() == 'y':
        services.append(setup_redis())
    
    # Create configuration
    create_config()
    create_env_file()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup complete! You can now use MemOS.")
        print("\nQuick start:")
        print("1. Edit .env file with your actual values")
        print("2. Run: python examples/simple_memos.py")
        print("3. Check GETTING_STARTED.md for more examples")
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")

if __name__ == "__main__":
    main()