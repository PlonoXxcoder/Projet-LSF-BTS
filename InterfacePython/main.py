import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QGraphicsOpacityEffect, QGraphicsDropShadowEffect)
from PySide6.QtGui import QFont, QColor, QPalette, QLinearGradient
from PySide6.QtCore import (Qt, QPropertyAnimation, QEasingCurve, QTimer)
from mainwindow import Ui_MainWindow  # Import de votre interface générée

class AppIntro(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chargement...")
        self.setFixedSize(800, 500)
        self.setStyleSheet("background-color: rgb(10, 32, 77);")
        
        # Layout principal
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Logo
        self.logo = QLabel(self)
        self.logo.setAlignment(Qt.AlignCenter)
        self.logo.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 64px;
                font-weight: bold;
                margin-top: 100px;
            }
        """)
        self.logo.setFont(QFont("Segoe UI", 48, QFont.Bold))
        self.logo.setText("TraductionLSF")
        
        # Effet de glow bleu
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(30)
        glow.setColor(QColor(100, 150, 255, 150))
        glow.setOffset(0, 0)
        self.logo.setGraphicsEffect(glow)
        
        # Barre de progression
        self.progress = QWidget(self)
        self.progress.setFixedSize(500, 8)
        self.progress.setStyleSheet("background-color: rgba(0, 0, 0, 0.3); border-radius: 4px;")
        
        self.progress_bar = QWidget(self.progress)
        self.progress_bar.setGeometry(0, 0, 0, 8)
        self.progress_bar.setStyleSheet("""
            background-color: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 rgb(50, 100, 200), stop:1 rgb(100, 150, 255)
            );
            border-radius: 4px;
        """)
        
        layout.addStretch()
        layout.addWidget(self.logo)
        layout.addSpacing(40)
        layout.addWidget(self.progress, alignment=Qt.AlignCenter)
        layout.addStretch()
        
        self.init_animations()
    
    def init_animations(self):
        # Animation d'apparition
        logo_opacity = QGraphicsOpacityEffect(self.logo)
        self.logo.setGraphicsEffect(logo_opacity)
        
        fade_in = QPropertyAnimation(logo_opacity, b"opacity")
        fade_in.setDuration(1500)
        fade_in.setStartValue(0)
        fade_in.setEndValue(1)
        
        # Animation de la barre
        self.progress_anim = QPropertyAnimation(self.progress_bar, b"minimumWidth")
        self.progress_anim.setDuration(2500)
        self.progress_anim.setStartValue(0)
        self.progress_anim.setEndValue(500)
        self.progress_anim.setEasingCurve(QEasingCurve.InOutQuad)
        
        fade_in.start()
        self.progress_anim.start()
        
        # Fermer après 3 secondes et lancer la fenêtre principale
        QTimer.singleShot(3000, self.launch_main_app)
    
    def launch_main_app(self):
        self.main_window = MainWindow()
        self.main_window.showMaximized()  # Affiche en plein écran
        self.close()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # Vous pouvez ajouter ici d'autres initialisations pour votre fenêtre principale

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Configuration du style avec vos couleurs
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 32, 77))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # Démarrer par l'intro
    intro = AppIntro()
    intro.show()
    
    sys.exit(app.exec())
