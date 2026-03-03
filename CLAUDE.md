# Titans PyTorch/MLX 项目

## 训练服务器

### H800 ST

| 项目 | 值 |
|------|-----|
| IP | 111.6.70.85 |
| SSH 端口 | 101 |
| 用户 | shuzuan |
| 密码 | Free2024 |

**硬件配置**:
- 主机名: xc2-ubuntu1
- 系统: Ubuntu，内核 6.8.0-90-generic
- GPU: 8x NVIDIA H800 (80GB HBM 每卡，共 640GB 显存)
- CUDA: 12.8（NVML 当前报驱动/内核不匹配，需重启解决，不影响文件操作）
- 内存: 1.5TB
- 磁盘: 2.0TB (`/dev/sda2`)，已用 651GB，剩余 1.3TB
- Python: 3.10.12

**SSH 接入**:
```bash
ssh -p 101 shuzuan@111.6.70.85
```

> 网络经腾讯云中转（本地 → 175.27.169.180:17890 → H800）
> SSH 保活参数: `-o ServerAliveInterval=60 -o ServerAliveCountMax=5`
