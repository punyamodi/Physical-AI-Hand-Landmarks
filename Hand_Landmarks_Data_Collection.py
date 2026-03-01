"""
Hand Landmarks Data Collection & Training Pipeline
--------------------------------------------------
A professional tool to collect, visualize, and train models on hand gesture landmarks.
Built for Physical AI applications and Human-Computer Interaction studies.

Author: Punyamodi (Enhanced by Antigravity)
License: MIT
"""

import sys
import os

# Add src to python path if needed (though local import should work)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui import HandCollectorApp

def main():
    print("🚀 Initializing Hand Landmarks Data Collection System...")
    print("💡 Tip: Use the GUI to record different gesture categories (Wave, Fist, Open, etc.)")
    print("📁 Data will be saved in the 'data/' directory as CSV files.")
    
    # Create app and run
    app = HandCollectorApp()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 System closed by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
