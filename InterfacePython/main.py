import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from mainwindow import Ui_MainWindow  # Import de la classe générée

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Charge l'interface Qt

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())