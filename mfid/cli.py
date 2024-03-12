import sys
import argparse
from PyQt5.QtWidgets import QApplication
from mfid.app.mfid_app import LauncherApp

def main():
    parser = argparse.ArgumentParser(description='MFID Application Launcher')
    parser.add_argument('command', nargs='?', help="the command to run (e.g., 'run')")
    args = parser.parse_args()

    if args.command == 'run':
        app = QApplication(sys.argv)
        ex = LauncherApp()
        ex.show()
        sys.exit(app.exec_())
    else:
        print("Unknown command. Use 'mfid run' to start the application.")

if __name__ == "__main__":
    main()
