import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the Streamlit app
from app.streamlit import main

if __name__ == "__main__":
    main() 