# 第 0 周：在 Windows 台式机上配环境（WSL2）

目标：在 WSL2 的 Ubuntu 里跑通 CUDA、PyTorch 和 Triton，然后能从 Mac 远程连过来写代码。

## 1. Windows 这一侧

1. 把 NVIDIA 驱动更新到最新版（用 NVIDIA App，或者去官网下载）。WSL 里的 GPU 用的就是 Windows 这边的驱动。
2. 以管理员身份打开 PowerShell，安装 WSL2 和 Ubuntu：

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   装完重启电脑。第一次打开 Ubuntu 时，设好用户名和密码。
3. 确认装的是 WSL2：运行 `wsl -l -v`，VERSION 那一栏应该是 2。

## 2. WSL（Ubuntu）这一侧

1. 先确认能看到 GPU：

   ```bash
   nvidia-smi
   ```

   能列出 RTX 5070 Ti，就说明驱动通了。**不要在 WSL 里再装 Linux 版的 NVIDIA 驱动。**

2. 安装 CUDA Toolkit。5070 Ti 是 sm_120，必须用 12.8 或更新的版本，旧版本识别不了这张卡。
   - 打开 NVIDIA 的 CUDA 下载页，依次选择 Linux → x86_64 → WSL-Ubuntu → 2.0 → deb (network)，照页面上给的命令一条条执行。
   - 最后一步只装 `cuda-toolkit-12-x`（x 用页面上给的版本号）。**不要装 `cuda` 或 `cuda-drivers` 这两个包**，它们会连驱动一起装。
   - 装完把 nvcc 加进 PATH：

   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   nvcc --version
   ```

3. 安装基础工具：

   ```bash
   sudo apt update && sudo apt install -y build-essential git python3-venv python3-dev
   ```

4. 给 GitHub 配一把 SSH key，然后 clone 这个仓库：

   ```bash
   ssh-keygen -t ed25519 -C "wan3@seas.upenn.edu"
   cat ~/.ssh/id_ed25519.pub
   ```

   - 把 `cat` 打出来的公钥复制下来，贴到 GitHub 的 Settings → SSH and GPG keys → New SSH key。
   - 然后测试连接，看到 `Hi trumpool!` 就说明通了，再 clone：

   ```bash
   ssh -T git@github.com
   git clone git@github.com:trumpool/gpu-inference-lab.git ~/gpu-inference-lab
   cd ~/gpu-inference-lab
   ```

   仓库要放在 WSL 自己的文件系统里（`~/` 下面），不要放在 `/mnt/c/`。跨系统读写会慢很多。

5. 建 Python 环境，装 PyTorch：

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install numpy
   ```

   `cu128` 是对应 CUDA 12.8 的版本。如果 pytorch.org 上有更新的 `cuXXX`，用新的也可以。

   `python3-dev` 是给 Triton 用的：它第一次运行时要编译一个 C 扩展，缺了会报 `Python.h: No such file or directory`。没装 `numpy` 的话，`import torch` 时会打一条警告。

6. 验证环境：

   ```bash
   python scripts/check_env.py
   cd cuda/00_hello && nvcc -arch=native hello.cu -o hello && ./hello && cd ../..
   ```

7. 试一下 Nsight Compute：

   ```bash
   ncu --version
   ncu ./cuda/00_hello/hello
   ```

   如果报 `ERR_NVGPUCTRPERM`（没有权限读 GPU 性能计数器），先记进 `notes/env.md`，等第 4 周要用 Nsight 时再处理。解决办法可能有两种：
   - 在 Windows 的 NVIDIA 控制面板里打开「允许所有用户访问 GPU 性能计数器」；
   - 给台式机装一个 Ubuntu 双系统。

## 3. 从 Mac 远程写代码（可选）

- 最省事的做法：直接在台式机上用 VS Code，加上 WSL 插件。
- 想在 Mac 上写的话：
  1. 在 Windows 设置里打开 OpenSSH 服务器；
  2. 在 Mac 上用 VS Code 或 Cursor 的 Remote-SSH 连到 Windows；
  3. 连上以后，用 WSL 插件进入 Ubuntu；
  4. 人不在家时，用 Tailscale 把 Mac 和台式机组进同一个网络。

## 4. 笔记本（3070 Laptop）

步骤和上面一样。笔记本的 GPU 有功耗墙，插不插电、开什么性能模式，测出来的速度会差很多。所以正式的 benchmark 一律在台式机上测。
