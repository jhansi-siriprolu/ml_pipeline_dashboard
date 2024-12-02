import subprocess
import os
import sys

def main():
    script_path = os.path.join(os.path.dirname(__file__), "app.py")
    try:
        # Use subprocess to invoke the Streamlit app
        subprocess.run(["streamlit", "run", script_path], check=True)
    except FileNotFoundError:
        sys.stderr.write("Error: Streamlit is not installed.\n")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Error running Streamlit app: {e}\n")
