# WhisperMeeting

Windows 本地 GPU 实时会议字幕。采集 WASAPI 输出音频，使用 faster-whisper Large-v3-turbo 转写。

## 功能
- 电脑音频回环采集，支持全部输出设备或指定设备。
- 实时临时字幕与定稿校正，仅显示原文。
- 深色桌面界面、可调字号、术语提示、TXT 导出。
- 自动保存会议原文与恢复用临时字幕；缓存位于程序目录。

## 运行
Windows x64、兼容 CUDA 12/cuDNN 9 的 NVIDIA 显卡。
程序包解压后，model-turbo 和 cuda 目录必须位于 WhisperMeeting.exe 同级。
点击开始并等待预热。所有依赖就位后识别不需联网。

## 从源码构建
Python 3.11，安装 requirements.txt 后运行 build.ps1。
模型：mobiuslabsgmbh/faster-whisper-large-v3-turbo。
GPU 运行库：NVIDIA cuBLAS 12、cuDNN 9（官方 Windows wheel 中的 DLL）。
把模型放在源码旁 model-turbo，运行库放在 cuda；开发运行 python realtime.py。

## 验证范围
本机约113秒连续回放，144次更新、12段定稿均保留；更新间隔约0.8秒，处理延迟P95约0.31秒。这不是任意会议或硬件的性能保证。
尚未进行数小时压力测试。口音、噪声、多路同时说话会影响识别。

## 数据与来源
程序运行时会在本地 sessions 保存字幕，在本地写入日志。发布包不包含开发者的会议记录、术语设置或日志。
本程序使用 faster-whisper、CTranslate2、PyAudioWPatch、Tkinter 和 Whisper 模型；最初基于 collabora/WhisperLive 部署衍生的桌面工作台。源码中 desktop.py 是早期界面的兼容基类，realtime.py 是正式入口。
第三方程序、模型与 GPU 运行库分别适用其原许可证。请保留随附的许可证文件。

## 官方依赖下载（本仓库不托管大模型及 CUDA 包）

- 模型发布方：[Large-v3-turbo CTranslate2 模型](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo/tree/main)。原 mobiuslabsgmbh 地址已重定向到该组织。
- NVIDIA cuBLAS Windows 包：[nvidia-cublas-cu12 12.9.2.10](https://pypi.org/project/nvidia-cublas-cu12/12.9.2.10/#files)。
- NVIDIA cuDNN Windows 包：[nvidia-cudnn-cu12 9.9.0.52](https://pypi.org/project/nvidia-cudnn-cu12/9.9.0.52/#files)。

下载模型文件放入 model-turbo；从 NVIDIA Windows wheel 解压 DLL 到 cuda。两个目录需放在 realtime.py 或 WhisperMeeting.exe 同级。

也可运行提供的脚本，从上述发布方直接下载（需要 Python 3.11）：

```powershell
py -3.11 -m pip install "huggingface-hub>=0.36,<1"
py -3.11 download_dependencies.py --destination .
```

脚本对 NVIDIA 文件验证官方 PyPI 提供的 SHA-256。模型较大，请准备足够空间与稳定网络。临时文件放在目标目录内。

本次 Release 先发布源码，未附 Windows EXE；使用 requirements.txt 安装依赖后执行 build.ps1 可自行构建。不要下载历史校验列表中提到、但本 Release 没有提供的附件。
