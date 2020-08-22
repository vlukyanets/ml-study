import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton


class DigitsRecognitionMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.button_size = 64
        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 400, 400)
        self.setWindowTitle("DR NN Test Window")
        layout = QGridLayout()

        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)

        self.buttons = []
        for i in range(8):
            self.buttons.append([])
            for j in range(8):
                button = QPushButton()
                button.setStyleSheet("background-color: white")
                button.setFixedSize(self.button_size, self.button_size)
                button.setEnabled(False)
                layout.addWidget(button, i, j)
                self.buttons[i].append(button)

        self.setLayout(layout)

    def set_image(self, image):
        for i in range(8):
            for j in range(8):
                color = image[i][j]
                self.buttons[i][j].setStyleSheet(
                    "background-color: rgb({0}, {0}, {0})".format(255 - 12 * color))


def display_digit(image):
    app = QApplication(sys.argv)
    window = DigitsRecognitionMainWindow()
    window.show()
    window.set_image(image)
    app.exec_()


if __name__ == "__main__":
    print("This module should not be imported")
