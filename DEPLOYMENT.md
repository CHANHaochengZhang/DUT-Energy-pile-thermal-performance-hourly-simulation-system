# 能量桩热-力性能模拟系统部署指南

## 🚀 快速部署方案

### 方案一：Streamlit Cloud（推荐）

**优点：**
- 完全免费
- 部署简单，5分钟完成
- 自动HTTPS
- 支持GitHub集成

**部署步骤：**

1. **准备GitHub仓库**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <你的GitHub仓库地址>
   git push -u origin main
   ```

2. **访问Streamlit Cloud**
   - 打开 [share.streamlit.io](https://share.streamlit.io)
   - 使用GitHub账号登录

3. **部署应用**
   - 选择你的GitHub仓库
   - 设置主文件路径：`mystreamlit.py`
   - 点击"Deploy"按钮

4. **等待部署完成**
   - 部署时间：2-5分钟
   - 获得公开访问链接

### 方案二：本地部署

**适用于：**
- 内网使用
- 需要自定义配置
- 短期演示

**部署步骤：**

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行应用**
   ```bash
   streamlit run mystreamlit.py --server.port 8501
   ```

3. **访问应用**
   - 本地：http://localhost:8501
   - 局域网：http://你的IP:8501

### 方案三：Docker部署

**适用于：**
- 生产环境
- 需要容器化
- 多环境部署

**部署步骤：**

1. **创建Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 8501
   
   CMD ["streamlit", "run", "mystreamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **构建和运行**
   ```bash
   docker build -t dutpowersim .
   docker run -p 8501:8501 dutpowersim
   ```

## 📋 部署前检查清单

- [ ] 所有依赖包已添加到 `requirements.txt`
- [ ] 图片文件路径正确（tech_road.jpg, HP-cop.jpg等）
- [ ] 数据文件存在（能量桩负荷数据.xlsx等）
- [ ] 模型文件完整（COP_heating.json, COP_colding.json等）

## 🔧 常见问题解决

### 1. 依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2. 端口被占用
```bash
# 查看端口占用
netstat -ano | findstr 8501

# 使用其他端口
streamlit run mystreamlit.py --server.port 8502
```

### 3. 图片显示问题
- 确保图片文件在正确路径
- 检查文件名大小写
- 使用相对路径

## 🌐 访问控制

### 公开访问
- 默认情况下，Streamlit Cloud部署的应用是公开的
- 任何人都可以访问和使用

### 限制访问（可选）
如果需要限制访问，可以考虑：
- 添加用户认证
- 使用VPN或内网部署
- 设置访问密码

## 📊 性能优化建议

1. **缓存计算结果**
   - 使用 `@st.cache_data` 装饰器
   - 避免重复计算

2. **减少内存使用**
   - 及时清理大型变量
   - 使用生成器处理大数据

3. **优化计算**
   - 使用向量化操作
   - 并行计算（已实现）

## 🔒 安全注意事项

1. **数据安全**
   - 不要上传敏感数据
   - 定期清理临时文件

2. **系统安全**
   - 使用最新版本的依赖包
   - 定期更新系统

## 📞 技术支持

如果遇到部署问题，请检查：
1. 错误日志信息
2. 依赖包版本兼容性
3. 系统环境要求

---

**推荐部署顺序：**
1. 先尝试Streamlit Cloud（最简单）
2. 如需内网使用，选择本地部署
3. 生产环境考虑Docker部署
