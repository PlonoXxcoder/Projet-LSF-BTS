# -*- coding: utf-8 -*-

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt, Signal)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QFrame, 
    QGridLayout, QHBoxLayout, QLabel, QLayout, 
    QMainWindow, QMenuBar, QPushButton, QSizePolicy, 
    QSpacerItem, QStatusBar, QTextEdit, QVBoxLayout, 
    QWidget)

class Ui_ParametersWindow(QWidget):
    color_changed = Signal(QColor)
    bg_color_changed = Signal(QColor)
    
    def setupUi(self, ParametersWindow):
        ParametersWindow.setObjectName(u"ParametersWindow")
        ParametersWindow.resize(400, 350)
        
        self.centralwidget = QWidget(ParametersWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        
        self.layout = QVBoxLayout(self.centralwidget)
        
        # Titre
        self.label = QLabel("Paramètres de l'application", self.centralwidget)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(u"font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        self.layout.addWidget(self.label)
        
        # Section Couleur du texte
        self.text_color_frame = QFrame()
        self.text_color_layout = QHBoxLayout(self.text_color_frame)
        
        self.text_color_label = QLabel("Couleur du texte:", self.centralwidget)
        self.text_color_btn = QPushButton("Choisir", self.centralwidget)
        self.text_color_btn.clicked.connect(self.choose_text_color)
        self.text_color_preview = QLabel()
        self.text_color_preview.setFixedSize(30, 30)
        self.text_color_preview.setStyleSheet("background-color: white; border: 1px solid black;")
        
        self.text_color_layout.addWidget(self.text_color_label)
        self.text_color_layout.addWidget(self.text_color_btn)
        self.text_color_layout.addWidget(self.text_color_preview)
        self.layout.addWidget(self.text_color_frame)
        
        # Section Couleur de fond
        self.bg_color_frame = QFrame()
        self.bg_color_layout = QHBoxLayout(self.bg_color_frame)
        
        self.bg_color_label = QLabel("Couleur de fond:", self.centralwidget)
        self.bg_color_btn = QPushButton("Choisir", self.centralwidget)
        self.bg_color_btn.clicked.connect(self.choose_bg_color)
        self.bg_color_preview = QLabel()
        self.bg_color_preview.setFixedSize(30, 30)
        self.bg_color_preview.setStyleSheet("background-color: rgb(10, 32, 77); border: 1px solid black;")
        
        self.bg_color_layout.addWidget(self.bg_color_label)
        self.bg_color_layout.addWidget(self.bg_color_btn)
        self.bg_color_layout.addWidget(self.bg_color_preview)
        self.layout.addWidget(self.bg_color_frame)
        
        # Boutons
        self.buttons_frame = QFrame()
        self.buttons_layout = QHBoxLayout(self.buttons_frame)
        
        self.default_btn = QPushButton("Par défaut", self.centralwidget)
        self.default_btn.clicked.connect(self.reset_defaults)
        self.close_btn = QPushButton("Fermer", self.centralwidget)
        self.close_btn.clicked.connect(ParametersWindow.close)
        
        self.buttons_layout.addWidget(self.default_btn)
        self.buttons_layout.addWidget(self.close_btn)
        self.layout.addWidget(self.buttons_frame)
        
        ParametersWindow.setCentralWidget(self.centralwidget)
    
    def choose_text_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.text_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
            self.color_changed.emit(color)
    
    def choose_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
            self.bg_color_changed.emit(color)
    
    def reset_defaults(self):
        default_text = QColor("white")
        default_bg = QColor(10, 32, 77)
        
        self.text_color_preview.setStyleSheet(f"background-color: {default_text.name()}; border: 1px solid black;")
        self.bg_color_preview.setStyleSheet(f"background-color: rgb(10, 32, 77); border: 1px solid black;")
        
        self.color_changed.emit(default_text)
        self.bg_color_changed.emit(default_bg)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(1195, 692)
        
        # Configuration initiale
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        MainWindow.setSizePolicy(sizePolicy)
        
        # Style par défaut
        self.default_bg_color = "rgb(10, 32, 77)"
        self.default_text_color = "white"
        
        # Widget central
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setStyleSheet(f"background-color: {self.default_bg_color}; color: {self.default_text_color};")
        self.gridLayout_3 = QGridLayout(self.centralwidget)
        
        # Barre d'outils supérieure
        self.setup_top_toolbar()
        
        # Vue caméra
        self.setup_camera_view()
        
        # Contrôles audio et zone de texte
        self.setup_audio_text_controls()
        
        # Configuration finale
        MainWindow.setCentralWidget(self.centralwidget)
        self.setup_menu_statusbar(MainWindow)
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)
        
        # Connecter le bouton paramètres
        self.boutonparametre.clicked.connect(self.open_parameters)
    
    def setup_top_toolbar(self):
        self.gridLayout = QGridLayout()
        
        # Bouton Paramètres
        self.boutonparametre = QPushButton(self.centralwidget)
        self.boutonparametre.setMinimumSize(QSize(70, 70))
        self.boutonparametre.setMaximumSize(QSize(70, 70))
        self.boutonparametre.setIcon(QIcon(":/new/prefix1/Parametre.png"))
        self.boutonparametre.setIconSize(QSize(30, 30))
        self.gridLayout.addWidget(self.boutonparametre, 1, 0, 1, 1)
        
        # Logo
        self.logo = QLabel(self.centralwidget)
        self.logo.setPixmap(QPixmap(":/new/prefix1/logolsf2.png"))
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLayout.addWidget(self.logo, 1, 1, 1, 1)
        
        self.gridLayout_3.addLayout(self.gridLayout, 2, 1, 1, 1)
    
    def setup_camera_view(self):
        self.gridLayout_2 = QGridLayout()
        self.frame = QFrame(self.centralwidget)
        
        # Configuration de la vue caméra
        self.camera_view = QLabel(self.frame)
        self.camera_view.setMinimumSize(QSize(300, 300))
        self.camera_view.setStyleSheet(u"""
            background-color: black;
            border: 2px solid #00aaff;
            border-radius: 5px;
        """)
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.gridLayout_5 = QGridLayout(self.frame)
        self.gridLayout_5.addWidget(self.camera_view, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 3, 1, 1, 1)
    
    def setup_audio_text_controls(self):
        self.horizontalLayout = QHBoxLayout()
        
        # Contrôles audio
        self.setup_audio_controls()
        
        # Zone de texte
        self.setup_text_area()
        
        # Bouton Exportation
        self.setup_export_button()
        
        self.gridLayout_3.addLayout(self.horizontalLayout, 4, 1, 1, 1)
    
    def setup_audio_controls(self):
        self.verticalLayout = QVBoxLayout()
        
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setText(u" Traduction : ")
        font = QFont()
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verticalLayout.addWidget(self.label_2)
        
        self.activesynthese = QPushButton(self.centralwidget)
        self.activesynthese.setMinimumSize(QSize(70, 70))
        self.activesynthese.setIcon(QIcon(":/new/prefix1/son.PNG"))
        self.activesynthese.setIconSize(QSize(30, 30))
        self.verticalLayout.addWidget(self.activesynthese)
        
        self.desactivesynthese = QPushButton(self.centralwidget)
        self.desactivesynthese.setMinimumSize(QSize(70, 70))
        self.desactivesynthese.setIcon(QIcon(":/new/prefix1/mute.PNG"))
        self.desactivesynthese.setIconSize(QSize(30, 30))
        self.verticalLayout.addWidget(self.desactivesynthese)
        
        self.horizontalLayout.addLayout(self.verticalLayout)
    
    def setup_text_area(self):
        self.gridLayout_4 = QGridLayout()
        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setMaximumSize(QSize(16777215, 200))
        
        self.textEdit = QTextEdit(self.frame_2)
        self.textEdit.setStyleSheet(u"font: 600 16pt \"Segoe UI\";")
        
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.addWidget(self.textEdit)
        self.gridLayout_4.addWidget(self.frame_2, 1, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_4)
    
    def setup_export_button(self):
        self.exportation = QPushButton(self.centralwidget)
        self.exportation.setMinimumSize(QSize(70, 70))
        self.exportation.setMaximumSize(QSize(80, 16777215))
        self.exportation.setIcon(QIcon(":/new/prefix1/exporter.PNG"))
        self.exportation.setIconSize(QSize(30, 30))
        self.horizontalLayout.addWidget(self.exportation)
    
    def setup_menu_statusbar(self, MainWindow):
        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 1195, 33))
        MainWindow.setMenuBar(self.menubar)
    
    def open_parameters(self):
        self.parameters_window = QMainWindow()
        self.ui_parameters = Ui_ParametersWindow()
        self.ui_parameters.setupUi(self.parameters_window)
        
        # Connecter les signaux
        self.ui_parameters.color_changed.connect(self.update_text_colors)
        self.ui_parameters.bg_color_changed.connect(self.update_bg_color)
        
        self.parameters_window.show()
    
    def update_text_colors(self, color):
        color_name = color.name()
        self.label_2.setStyleSheet(f"color: {color_name}; font-weight: bold;")
        self.textEdit.setStyleSheet(f"font: 600 16pt \"Segoe UI\"; color: {color_name};")
    
    def update_bg_color(self, color):
        style = f"background-color: {color.name()}; color: {self.default_text_color};"
        self.centralwidget.setStyleSheet(style)
    
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"TraductionLSF", None))
        self.boutonparametre.setText("")
        self.logo.setText("")
        self.camera_view.setText("")
        self.label_2.setText(QCoreApplication.translate("MainWindow", u" Traduction : ", None))
        self.activesynthese.setText("")
        self.desactivesynthese.setText("")
        self.exportation.setText("")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
