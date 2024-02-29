import sys
from PyQt5.QtWidgets import QApplication
from mfid.app.mfid_app import LauncherApp

def main():
    app = QApplication(sys.argv)
    ex = LauncherApp()
    ex.show()
    sys.exit(app.exec_())
