#!/usr/bin/env python3
"""
Interactive Setup Script for Multi-Modal RAG + Training Pipeline
Prompts user for configuration and automatically sets up the environment.
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path
import json

def run_command(command, check=True, capture_output=False):
    """Run a shell command and return the result"""
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(command, shell=True, check=check)
            return result.returncode, "", ""
    except subprocess.CalledProcessError as e:
        if capture_output:
            return e.returncode, e.stdout, e.stderr
        else:
            return e.returncode, "", ""

def check_postgresql_installed():
    """Check if PostgreSQL is installed and running"""
    print("🔍 Checking PostgreSQL installation...")
    
    # Check if PostgreSQL is installed
    returncode, stdout, stderr = run_command("which psql", capture_output=True)
    if returncode != 0:
        print("❌ PostgreSQL is not installed. Please run the install.sh script first.")
        return False
    
    # Check if PostgreSQL service is running
    returncode, stdout, stderr = run_command("systemctl is-active --quiet postgresql", capture_output=True)
    if returncode != 0:
        print("⚠️  PostgreSQL service is not running. Attempting to start it...")
        run_command("sudo systemctl start postgresql")
        
        # Check again
        returncode, stdout, stderr = run_command("systemctl is-active --quiet postgresql", capture_output=True)
        if returncode != 0:
            print("❌ Failed to start PostgreSQL service. Please start it manually:")
            print("   sudo systemctl start postgresql")
            return False
    
    print("✅ PostgreSQL is installed and running")
    return True

def create_postgresql_user(username, password):
    """Create a PostgreSQL user with the specified credentials"""
    print(f"👤 Creating PostgreSQL user '{username}'...")
    
    # Create user with password
    create_user_cmd = f"sudo -u postgres psql -c \"CREATE USER {username} WITH PASSWORD '{password}';\""
    returncode, stdout, stderr = run_command(create_user_cmd, capture_output=True)
    
    if returncode != 0:
        if "already exists" in stderr:
            print(f"⚠️  User '{username}' already exists. Updating password...")
            update_password_cmd = f"sudo -u postgres psql -c \"ALTER USER {username} WITH PASSWORD '{password}';\""
            returncode, stdout, stderr = run_command(update_password_cmd, capture_output=True)
            if returncode != 0:
                print(f"❌ Failed to update password for user '{username}': {stderr}")
                return False
        else:
            print(f"❌ Failed to create user '{username}': {stderr}")
            return False
    
    print(f"✅ PostgreSQL user '{username}' created/updated successfully")
    return True

def create_postgresql_database(dbname, username):
    """Create a PostgreSQL database and grant privileges"""
    print(f"🗄️  Creating PostgreSQL database '{dbname}'...")
    
    # Create database
    create_db_cmd = f"sudo -u postgres createdb {dbname}"
    returncode, stdout, stderr = run_command(create_db_cmd, capture_output=True)
    
    if returncode != 0:
        if "already exists" in stderr:
            print(f"⚠️  Database '{dbname}' already exists")
        else:
            print(f"❌ Failed to create database '{dbname}': {stderr}")
            return False
    
    # Grant privileges
    grant_cmd = f"sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE {dbname} TO {username};\""
    returncode, stdout, stderr = run_command(grant_cmd, capture_output=True)
    
    if returncode != 0:
        print(f"❌ Failed to grant privileges: {stderr}")
        return False
    
    print(f"✅ Database '{dbname}' created and privileges granted")
    return True

def setup_pgvector_extension(dbname):
    """Setup pgvector extension in the database"""
    print("🔧 Setting up pgvector extension...")
    
    # Enable pgvector extension
    enable_cmd = f"sudo -u postgres psql -d {dbname} -c \"CREATE EXTENSION IF NOT EXISTS vector;\""
    returncode, stdout, stderr = run_command(enable_cmd, capture_output=True)
    
    if returncode != 0:
        print(f"❌ Failed to enable pgvector extension: {stderr}")
        print("   Please ensure pgvector is installed: sudo apt install postgresql-XX-pgvector")
        return False
    
    print("✅ pgvector extension enabled successfully")
    return True

def get_user_input(prompt, default=None, password=False):
    """Get user input with optional default value and password masking"""
    if default:
        prompt = f"{prompt} (default: {default}): "
    else:
        prompt = f"{prompt}: "
    
    if password:
        value = getpass.getpass(prompt)
    else:
        value = input(prompt).strip()
    
    if not value and default:
        return default
    return value

def create_env_file(config):
    """Create the .env file with the provided configuration"""
    print("📝 Creating .env file...")
    
    env_content = f"""# Database Configuration
DB_HOST={config['db_host']}
DB_PORT={config['db_port']}
DB_NAME={config['db_name']}
DB_USER={config['db_user']}
DB_PASSWORD={config['db_password']}

# Model Configuration
BASE_MODEL={config['base_model']}
USE_GPU={config['use_gpu']}

# Training Configuration
EPOCHS={config['epochs']}
BATCH_SIZE={config['batch_size']}
LEARNING_RATE={config['learning_rate']}
MAX_LENGTH={config['max_length']}

# Pipeline Configuration
INPUT_DIR={config['input_dir']}
OUTPUT_DIR={config['output_dir']}
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created successfully")

def create_config_json(config):
    """Create the config.json file with the provided configuration"""
    print("📝 Creating config.json file...")
    
    config_json = {
        "input_dir": config['input_dir'],
        "extracted_assets_dir": f"{config['output_dir']}/extracted_assets",
        "training_data_dir": f"{config['output_dir']}/training_data",
        "fine_tuned_models_dir": f"{config['output_dir']}/fine_tuned_models",
        "database_config": {
            "host": config['db_host'],
            "port": config['db_port'],
            "database": config['db_name'],
            "user": config['db_user'],
            "password": config['db_password']
        },
        "base_model": config['base_model'],
        "epochs": config['epochs'],
        "batch_size": config['batch_size'],
        "learning_rate": config['learning_rate'],
        "use_gpu": config['use_gpu'],
        "output_format": "jsonl"
    }
    
    with open('config.json', 'w') as f:
        json.dump(config_json, f, indent=2)
    
    print("✅ config.json file created successfully")

def main():
    """Main setup function"""
    print("🚀 Multi-Modal RAG + Training Pipeline Setup")
    print("=" * 50)
    
    # Check if PostgreSQL is installed and running
    if not check_postgresql_installed():
        sys.exit(1)
    
    print("\n📋 Configuration Setup")
    print("-" * 30)
    
    # Get database configuration
    print("\n🗄️  Database Configuration:")
    db_host = get_user_input("Database host", "localhost")
    db_port = get_user_input("Database port", "5432")
    db_name = get_user_input("Database name", "multimodal_rag")
    db_user = get_user_input("Database username", "multimodal_user")
    db_password = get_user_input("Database password", password=True)
    
    # Get model configuration
    print("\n🤖 Model Configuration:")
    base_model = get_user_input("Base model for fine-tuning", "llama2")
    use_gpu = get_user_input("Use GPU for training (true/false)", "true").lower() == "true"
    
    # Get training configuration
    print("\n🎯 Training Configuration:")
    epochs = int(get_user_input("Number of training epochs", "3"))
    batch_size = int(get_user_input("Training batch size", "4"))
    learning_rate = float(get_user_input("Learning rate", "2e-5"))
    max_length = int(get_user_input("Maximum sequence length", "512"))
    
    # Get pipeline configuration
    print("\n📁 Pipeline Configuration:")
    input_dir = get_user_input("Input directory for PDFs", "data")
    output_dir = get_user_input("Output directory for pipeline results", "pipeline_output")
    
    # Create configuration dictionary
    config = {
        'db_host': db_host,
        'db_port': db_port,
        'db_name': db_name,
        'db_user': db_user,
        'db_password': db_password,
        'base_model': base_model,
        'use_gpu': use_gpu,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'input_dir': input_dir,
        'output_dir': output_dir
    }
    
    print("\n🔧 Setting up PostgreSQL...")
    
    # Create PostgreSQL user
    if not create_postgresql_user(db_user, db_password):
        print("❌ Failed to create PostgreSQL user. Setup aborted.")
        sys.exit(1)
    
    # Create PostgreSQL database
    if not create_postgresql_database(db_name, db_user):
        print("❌ Failed to create PostgreSQL database. Setup aborted.")
        sys.exit(1)
    
    # Setup pgvector extension
    if not setup_pgvector_extension(db_name):
        print("❌ Failed to setup pgvector extension. Setup aborted.")
        sys.exit(1)
    
    print("\n📝 Creating configuration files...")
    
    # Create .env file
    create_env_file(config)
    
    # Create config.json file
    create_config_json(config)
    
    # Create directories
    print("\n📁 Creating directories...")
    directories = [input_dir, output_dir, f"{output_dir}/extracted_assets", 
                   f"{output_dir}/training_data", f"{output_dir}/fine_tuned_models", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Summary:")
    print(f"   Database: {db_name} on {db_host}:{db_port}")
    print(f"   User: {db_user}")
    print(f"   Base model: {base_model}")
    print(f"   Input directory: {input_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Configuration files: .env, config.json")
    
    print("\n🚀 Next steps:")
    print("   1. Place your PDF files in the 'data' directory")
    print("   2. Run the pipeline: python run_pipeline.py")
    print("   3. Or run individual phases: python run_pipeline.py --phase 1")
    print("   4. Launch Streamlit UI: streamlit run scripts/streamlit_app.py")

if __name__ == "__main__":
    main() 