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

最省事的做法是直接在台式机上用 VS Code，加上 WSL 插件。想在 Mac 上写的话，链路是这样的：

```
Mac 上的 VS Code ──SSH──► Windows（OpenSSH 服务器）──► WSL 里的 Ubuntu
     Remote-SSH 插件                               WSL 插件
```

（2026-10-31 在台式机上配通。下面的命令除了标明在 Mac 上跑的，都在 **Windows 的管理员 PowerShell** 里跑。）

### 3.1 Windows：装 OpenSSH 服务器

文档里的标准做法是：

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd; Set-Service -Name sshd -StartupType Automatic
```

台式机上这一步报了 `0x800f0993`。看 `C:\Windows\Logs\DISM\dism.log`，文件下载下来了，是在最后 `Failed finalizing changes` 那步失败的，当时系统里有个更新在等重启（CBS `RebootPending`）。最后改用 winget 装微软在 GitHub 上维护的 Win32-OpenSSH，一次成功：

```powershell
winget install Microsoft.OpenSSH.Preview
```

装完 `sshd` 服务会自动运行、开机自启，程序在 `C:\Program Files\OpenSSH\`，配置在 `C:\ProgramData\ssh\`，还会建一条只对**专用网络**生效的防火墙规则。检查：

```powershell
Get-Service sshd
Get-NetFirewallRule | Where-Object DisplayName -match ssh | Select-Object DisplayName, Enabled, Profile
```

### 3.2 Windows：把网络设成专用网络

防火墙规则只对专用网络放行 22 端口。家里的 Wi-Fi 如果被设成了"公用网络"，Mac 的连接会被直接挡掉。在"设置 → 网络和 Internet → Wi-Fi → 当前网络"里改成**专用网络**。不要反过来让规则对公用网络放行，否则在外面连公共 Wi-Fi 时 22 端口也开着。

### 3.3 放 Mac 的公钥（这里最容易出错）

在 Mac 上（没有 key 就先 `ssh-keygen -t ed25519 -C "mac"`）：

```bash
cat ~/.ssh/id_ed25519.pub
```

**Windows 账号如果在管理员组里**，sshd 不读 `C:\Users\<你>\.ssh\authorized_keys`，只读 `C:\ProgramData\ssh\administrators_authorized_keys`（`sshd_config` 末尾的 `Match Group administrators` 写的就是这个），而且这个文件只能让管理员和 SYSTEM 访问：

```powershell
notepad C:\ProgramData\ssh\administrators_authorized_keys
# 把 Mac 的公钥那一整行贴进去保存
icacls.exe "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
```

**坑：记事本会偷偷加 `.txt` 后缀。** 台式机上第一次就存成了 `administrators_authorized_keys.txt`，结果 sshd 找不到公钥，一直要求输密码。用 `Get-ChildItem C:\ProgramData\ssh` 确认文件名，不对就：

```powershell
Rename-Item C:\ProgramData\ssh\administrators_authorized_keys.txt administrators_authorized_keys
```

第一次从 Mac 连的时候会让你确认服务器指纹，可以先在 Windows 上看一眼再回 `yes`：

```powershell
& 'C:\Program Files\OpenSSH\ssh-keygen.exe' -lf C:\ProgramData\ssh\ssh_host_ed25519_key.pub
```

公钥登录通了以后，把 `C:\ProgramData\ssh\sshd_config` 里的 `#PasswordAuthentication yes` 改成 `PasswordAuthentication no`，再 `Restart-Service sshd`，关掉密码登录。改之前留一个已经连着的窗口，改完先开新终端测一下还能不能连。

### 3.4 Mac：SSH 配置和 VS Code

`~/.ssh/config`（`192.168.1.160` 是台式机在家里局域网的地址，只有 Mac 也连着家里的网络时才能用；人在外面见 3.5）：

```
Host jeff
    HostName 192.168.1.160
    User 24403
    IdentityFile ~/.ssh/id_ed25519

# 只给终端用：ssh jeff-wsl 一步进 Ubuntu。VS Code 的 Remote-SSH 处理不好 RemoteCommand，VS Code 里连 jeff
Host jeff-wsl
    HostName 192.168.1.160
    User 24403
    IdentityFile ~/.ssh/id_ed25519
    RequestTTY yes
    RemoteCommand wsl
```

VS Code：

1. 装 **Remote - SSH** 和 **WSL** 两个插件（或者直接装 Remote Development 插件包）。
2. `Cmd+Shift+P` → `Remote-SSH: Connect to Host...` → `jeff`，远端系统选 **Windows**。
3. 在新窗口里 `Cmd+Shift+P` → `WSL: Connect to WSL using Distro...` → `Ubuntu-24.04`。搜不到这个命令的话，先在插件栏把 WSL 插件装到 `SSH: jeff` 上。
4. 打开 `/home/wan/gpu-inference-lab`，终端里跑 `nvidia-smi` 能看到 5070 Ti 就说明整条链路通了。

只用终端的话：`ssh -t jeff wsl`。

### 3.5 其他

- **台式机别睡眠**：设置 → 系统 → 电源，接通电源时进入睡眠改成"从不"，否则人一走开就连不上。
- **局域网 IP 会变**：`192.168.1.160` 是路由器自动分配的。可以在路由器里做 DHCP 地址保留，或者直接用 Tailscale。
- **人不在家时**：用 Tailscale 把 Mac 和台式机组进同一个私人网络，不要在路由器上把 22 端口转发到公网。

## 4. 笔记本（3070 Laptop）

步骤和上面一样。笔记本的 GPU 有功耗墙，插不插电、开什么性能模式，测出来的速度会差很多。所以正式的 benchmark 一律在台式机上测。
