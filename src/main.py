"""
This is the main script that will run the GUI 
for the generation of spectra data, lookuptables 
and fitting of the experimental data.
"""

# Libraries
import sys
from PyQt5.QtWidgets import QApplication

# Importing the GUI script
from GUI_gen import GUI

def main():
    """
    Main function to run the GUI application.
    """
    # Create the application instance
    app = QApplication(sys.argv)
    
    # Create the main window instance
    window = GUI()
    
    # Show the main window
    window.show()
    
    # Execute the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()