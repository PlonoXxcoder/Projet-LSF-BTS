# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindowglDWBs.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLayout, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QSpacerItem, QStatusBar,
    QTextEdit, QVBoxLayout, QWidget)
import ressource_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(1195, 692)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        icon = QIcon()
        icon.addFile(u":/new/prefix1/logolsf2.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(u"background-color: rgb(10, 32, 77); color: white;")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setEnabled(True)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setAutoFillBackground(False)
        self.gridLayout_3 = QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.boutonparametre = QPushButton(self.centralwidget)
        self.boutonparametre.setObjectName(u"boutonparametre")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.boutonparametre.sizePolicy().hasHeightForWidth())
        self.boutonparametre.setSizePolicy(sizePolicy1)
        self.boutonparametre.setMinimumSize(QSize(70, 70))
        self.boutonparametre.setMaximumSize(QSize(70, 70))
        icon1 = QIcon()
        icon1.addFile(u":/new/prefix1/Parametre.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.boutonparametre.setIcon(icon1)
        self.boutonparametre.setIconSize(QSize(30, 30))

        self.gridLayout.addWidget(self.boutonparametre, 1, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 2, 1, 1)

        self.logo = QLabel(self.centralwidget)
        self.logo.setObjectName(u"logo")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.logo.sizePolicy().hasHeightForWidth())
        self.logo.setSizePolicy(sizePolicy2)
        self.logo.setMinimumSize(QSize(0, 0))
        self.logo.setPixmap(QPixmap(u":/new/prefix1/logolsf2.png"))
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.logo, 1, 1, 1, 1)


        self.gridLayout_3.addLayout(self.gridLayout, 2, 1, 1, 1)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy3)
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_5 = QGridLayout(self.frame)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.pushButton_2 = QPushButton(self.frame)
        self.pushButton_2.setObjectName(u"pushButton_2")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy4)
        self.pushButton_2.setMinimumSize(QSize(300, 300))
        self.pushButton_2.setStyleSheet(u"background-color: rgb(139, 139, 139);")
        icon2 = QIcon(QIcon.fromTheme(u"camera-video"))
        self.pushButton_2.setIcon(icon2)
        self.pushButton_2.setIconSize(QSize(30, 30))

        self.gridLayout_5.addWidget(self.pushButton_2, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)


        self.gridLayout_3.addLayout(self.gridLayout_2, 3, 1, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy5)
        font = QFont()
        font.setBold(True)
        font.setUnderline(False)
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.label_2)

        self.activesynthese = QPushButton(self.centralwidget)
        self.activesynthese.setObjectName(u"activesynthese")
        sizePolicy5.setHeightForWidth(self.activesynthese.sizePolicy().hasHeightForWidth())
        self.activesynthese.setSizePolicy(sizePolicy5)
        self.activesynthese.setMinimumSize(QSize(70, 70))
        self.activesynthese.setSizeIncrement(QSize(0, 0))
        self.activesynthese.setBaseSize(QSize(0, 0))
        self.activesynthese.setStyleSheet(u"background-color: rgb(10, 32, 77);")
        icon3 = QIcon()
        icon3.addFile(u":/new/prefix1/son.PNG", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.activesynthese.setIcon(icon3)
        self.activesynthese.setIconSize(QSize(30, 30))

        self.verticalLayout.addWidget(self.activesynthese)

        self.desactivesynthese = QPushButton(self.centralwidget)
        self.desactivesynthese.setObjectName(u"desactivesynthese")
        sizePolicy5.setHeightForWidth(self.desactivesynthese.sizePolicy().hasHeightForWidth())
        self.desactivesynthese.setSizePolicy(sizePolicy5)
        self.desactivesynthese.setMinimumSize(QSize(70, 70))
        self.desactivesynthese.setStyleSheet(u"background-color: rgb(10, 32, 77);")
        icon4 = QIcon()
        icon4.addFile(u":/new/prefix1/mute.PNG", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.desactivesynthese.setIcon(icon4)
        self.desactivesynthese.setIconSize(QSize(30, 30))
        self.desactivesynthese.setCheckable(False)

        self.verticalLayout.addWidget(self.desactivesynthese)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy6)
        self.frame_2.setMaximumSize(QSize(16777215, 200))
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.textEdit = QTextEdit(self.frame_2)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setStyleSheet(u"font: 600 16pt \"Segoe UI\";")

        self.horizontalLayout_2.addWidget(self.textEdit)


        self.gridLayout_4.addWidget(self.frame_2, 1, 0, 1, 1)


        self.horizontalLayout.addLayout(self.gridLayout_4)

        self.exportation = QPushButton(self.centralwidget)
        self.exportation.setObjectName(u"exportation")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.exportation.sizePolicy().hasHeightForWidth())
        self.exportation.setSizePolicy(sizePolicy7)
        self.exportation.setMinimumSize(QSize(70, 70))
        self.exportation.setMaximumSize(QSize(80, 16777215))
        icon5 = QIcon()
        icon5.addFile(u":/new/prefix1/exporter.PNG", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.exportation.setIcon(icon5)
        self.exportation.setIconSize(QSize(30, 30))

        self.horizontalLayout.addWidget(self.exportation)


        self.gridLayout_3.addLayout(self.horizontalLayout, 4, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1195, 33))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"TraductionLSF", None))
        self.boutonparametre.setText("")
        self.logo.setText("")
        self.pushButton_2.setText("")
        self.label_2.setText(QCoreApplication.translate("MainWindow", u" Traduction : ", None))
        self.activesynthese.setText("")
        self.desactivesynthese.setText("")
        self.exportation.setText("")
    # retranslateUi
