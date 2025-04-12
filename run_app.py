import streamlit.web.cli as stcli
import sys
import os

os.chdir("app")  # Переход в директорию app
sys.argv = ["streamlit", "run", "ui/app.py"]
sys.exit(stcli.main())
