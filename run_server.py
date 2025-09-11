"""
Simple server runner script for the Multi-Language Translation Platform
Supports English, Assamese, Bengali, Manipuri, and Santali
"""

import sys
import os
import subprocess

def main():
    """Run the FastAPI server"""
    # Add src directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    sys.path.insert(0, src_dir)
    
    # Change to the project directory
    os.chdir(current_dir)
    
    print("🌐 Starting Multi-Language Translation Platform...")
    print("🗣️  Supported Languages: English, Assamese, Bengali, Manipuri, Santali")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Run the FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\n💡 Make sure you have installed the requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
