from PyQt5.QtWidgets import QApplication
import sys
import srcTest

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = srcTest.ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
