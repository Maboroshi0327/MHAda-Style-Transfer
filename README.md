# 使用 Docker 執行 MHAda-Style-Transfer

本專案提供可重現的 Docker 執行環境，包含：
- Ubuntu 24.04 + Miniconda（Python 3.12）
- PyTorch 2.5.1 / CUDA 12.4（torchvision 0.20.1, torchaudio 2.5.1）
- 常用科學計算與影像套件（scipy, opencv-contrib-python, seaborn, imageio, imageio-ffmpeg）

相關檔案：
- Dockerfile: [Dockerfile](Dockerfile)
- 建置腳本: [build.sh](build.sh)
- 建立容器腳本: [create.sh](create.sh)

容器中的目錄規劃：
- 專案目錄：`/root/project`（對應到 MHAdaSTr）
- 資料集目錄：`/root/datasets`（對應到你的本機資料集路徑）

---

## 先決條件

- 已安裝 Docker（並可使用 `docker` 指令）
  - 安裝指引（Docker Engine）：https://docs.docker.com/engine/install/
  - Linux 後續設定（無 sudo 執行 docker 等）：https://docs.docker.com/engine/install/linux-postinstall/
  - GPU 參數說明（--gpus）：https://docs.docker.com/config/containers/resource_constraints/#gpu

- 需要 NVIDIA 顯示卡與驅動，以及 NVIDIA Container Toolkit 以啟用 `--gpus all`
  - 安裝 NVIDIA Container Toolkit：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  - 在容器中驗證 GPU 可用性：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#testing-the-nvidia-container-toolkit

- 本機準備好兩個目錄（將會掛載進容器）
  - 專案目錄（MHAdaSTr 目錄）
  - 資料集目錄（實際資料集放置處）

---

## 建置映像

你可以使用已提供的建置腳本或直接使用 docker 指令。

### 方式一：使用腳本建置
執行根目錄的 [build.sh](build.sh)，依提示輸入映像 Tag：
```
bash ./build.sh
# Tag name: mhada:latest
```

### 方式二：直接使用 docker 指令
```
docker build -t mhada:latest .
```

### 方式三：從 Docker Hub 取得預建映像
若不想本機建置，可直接拉取預建映像：
```
docker pull maboroshi327/vistytr:latest
```
注意：
- 後續指令中的映像名稱請使用 maboroshi327/vistytr:latest。
- 仍需於主機安裝 NVIDIA Container Toolkit 才能使用 `--gpus all`。

---

## 建立與掛載容器

容器會啟用 GPU，並將本機的專案與資料集目錄掛載到容器的 `/root/project` 與 `/root/datasets`。

### 方式一：使用腳本建立
執行 [create.sh](create.sh)，依提示輸入：
```
bash ./create.sh
# Container name: mhada
# Project mount path: /path/to/your/MHAdaSTr
# Datasets mount path: /path/to/your/datasets
# Image tag: mhada:latest
```

### 方式二：直接使用 docker 指令
```
CONTAINER_NAME=mhada
PROJECT_PATH=/path/to/your/MHAdaSTr
DATASETS_PATH=/path/to/your/datasets
TAG=maboroshi327/vistytr:latest  # 若你是本機建置，改成 mhada:latest

docker create --name $CONTAINER_NAME --ipc host -it --gpus all \
  -v $PROJECT_PATH:/root/project \
  -v $DATASETS_PATH:/root/datasets \
  $TAG
```

說明：
- `--gpus all` 需要 NVIDIA Container Toolkit
- `--ipc host` 可避免部分深度學習任務的共享記憶體限制
- 工作目錄預設為 `/root/project`（已在 [Dockerfile](Dockerfile) 中設定）。

---

## 啟動與進入容器（VS Code）

請先安裝 VS Code 擴充功能：
- Container Tools：https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-container-tools
- Dev Containers：https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers

接着使用 Dev Containers 進入容器內開發
- 點選遠端總管圖示
- 在選單中選擇「開發人員容器」
- 在列出的容器中，選擇要進入的容器，點選「在目前的視窗中附加」或「在新視窗中連結」
- 進入容器後，VS Code 會開啟 `/root/project` 目錄，在左側的檔案總管中能看到 Python 程式檔案

---

## 驗證環境與 GPU

進入容器後可執行：

- 檢查 GPU：
```
nvidia-smi
```

- 檢查 PyTorch CUDA：
```
/root/miniconda3/bin/python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

---

## 常見工作流程

以下為專案主要程式與用途：
- [train_image.py](MHAdaSTr/train_image.py)：訓練圖片風格轉換模型，輸出練後的權重檔。
- [train_video.py](MHAdaSTr/train_video.py)：訓練影片風格轉換模型，輸出訓練後的權重檔。
- [infer_image.py](MHAdaSTr/infer_image.py)：對單張圖片進行風格轉換推論，輸入內容圖片與風格圖片，輸出風格化圖片。
- [infer_video.py](MHAdaSTr/infer_video.py)：對影片進行風格轉換推論，輸入內容視訊與風格圖片，輸出結果視訊。

---

## 疑難排解

- 無法使用 GPU：
  - 確認主機已安裝對應 NVIDIA 驅動
  - 安裝 NVIDIA Container Toolkit，並確保 `docker run ... --gpus all` 可用
  - 在容器內執行 `nvidia-smi` 與上述 PyTorch 測試
  - 如果遇到突然無法使用 GPU 的情況，通常重啟容器後能解決

- Volume 權限或路徑錯誤：
  - 確認本機路徑存在且具有讀寫權限
  - 路徑中避免包含空白或特殊字元，或以引號包住路徑

---

## 參考檔案

- 建置步驟與內容：[Dockerfile](Dockerfile)
- 建置映像腳本：[build.sh](build.sh)
- 建立容器腳本：[create.sh](create.sh)