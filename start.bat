@echo off
echo 启动能量桩热-力性能模拟系统...
echo.
echo 正在检查Python环境...
python --version
echo.
echo 正在安装依赖包...
pip install -r requirements.txt
echo.
echo 启动Streamlit应用...
echo 应用将在浏览器中自动打开
echo 如果没有自动打开，请访问: http://localhost:8501
echo.
echo 按 Ctrl+C 停止应用
echo.
streamlit run mystreamlit.py --server.port 8501
pause
