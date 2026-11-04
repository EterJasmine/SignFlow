# SignFlow 手语知音

基于视觉感知、示范学习与反馈决策的手语练习智能体。交互角色为“手语小助手”。

本仓库从原手势舞项目中独立整理手语完整版：保留摄像头识别、视频分段标注、示范预览、纠错热更新、后台训练、小助手反馈与历史回顾。原仓库不受影响。

## 能力与边界

- 教程视频或摄像头录制进入同一标注流程，一视频支持多个词。
- MediaPipe检测、手部追踪、局部重检、特征提取与DTW同词多示范匹配。
- 动作诊断回放、用户纠正、纠错资料热更新。
- PyTorch GRU后台实验训练、手动重新训练及验证报告。
- DeepSeek根据检测事实生成小助手反馈与历史总结；未配置时回退为规则反馈。

系统主要检索已经收录的有限词汇，不是任意中国手语翻译器。匹配距离不是词义概率或规范性评分，实验分类器不自动替换线上算法。

## 环境与安装

当前开发环境使用Python 3.13和Node.js 24。推荐先在本机运行；Linux容器配置已准备，但未完成完整容器构建验证。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe scripts/download_models.py
npm --prefix apps/web ci
npm --prefix apps/web run build
```

将 `.env.example` 复制为 `.env`，需要大模型讲解时填写自己的 `DEEPSEEK_API_KEY`、账号可用的 `DEEPSEEK_MODEL` 与对应官方接口地址。不要提交真实密钥。没有密钥时基础识别和规则小助手仍可运行。

## 启动

```powershell
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

也可双击 `启动SignFlow.cmd`。访问 http://127.0.0.1:8000/ 。若原项目已经占用8000端口，可将新项目改用8001，不必停止原服务。

## 第一次使用

1. 上传或录制一个清晰完整的示范。
2. 设置动作起止时间，填写词义并保存。
3. 点击预览确认区间。一个视频可以继续标注多个词。
4. 用另一次独立录制进行识别，或在小助手中选择目标练习。
5. 识别错误时，先预览对应动作，再确认含义并纠正。
6. 合格纠错资料立即用于后续匹配，后台另行执行实验训练。

新克隆仓库默认没有示范库，不能直接识别原项目中的私人词库。

## 结构

```text
apps/api/              FastAPI接口与任务管理
apps/web/              手语前端
packages/pipeline/     检测 追踪 特征 匹配 小助手
training/              公共训练组件
scripts/               下载模型 训练与离线评估
tests/                自动测试
evaluation/           评估工具 不含个人真值文件
docs/                 项目说明与使用说明
deploy/               部署准备文件
```

视频分析仍调用原项目公共追踪、分段、图解与版本组件，因此相关底层代码保留用于兼容；原手势舞前端及歌词、教程导出的独立路由入口不作为本项目功能提供。文档中的开发历史和数据快照用于说明项目过程，不是随仓库分发的数据。

## 测试与训练

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp
.\.venv\Scripts\python.exe scripts/train_sign_temporal.py
```

也可在示范库点击“重新训练”。训练需本地有效示范及可隔离验证的类别；空库无法训练。训练集高分不能替代跨来源验证。离线评估脚本可能需要自行建立评价标签文件。

## 数据与配置

不上传 `.env`、数据库、摄像头录制、用户视频、通用模型二进制、训练权重与运行日志。运行资料位于 `data/`，模型位于 `models/`，实验权重与报告位于 `training/runs/`。

首次安装需自行下载模型，配置凭据并建立示范库。请备份数据目录，不要依赖Git保存私人资料。

当前实时引擎面向本地单用户。多人公网使用还需要会话隔离和访问控制。浏览器静态独立站是另一实现，不等于本仓库Python完整版。

## 文档

- [项目说明](docs/手语知音项目说明文档.md)
- [使用说明](docs/手语知音使用说明.md)

通用模型和参考实现的许可由原作者规定；本仓库未替第三方代码或模型添加授权声明。
