# WatchMyBirds - Real-time object detection and classification

[![Build and Push Docker Image](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml/badge.svg)](https://github.com/arminfabritzek/WatchMyBirds/actions/workflows/docker.yml) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)


![WatchMyBirds in Action](assets/birds_1280.gif)
*Real-time object detection and classification in action!*

---

![WatchMyBirds in Action](assets/app_landing_page.jpg)
*Clean, user-friendly web interface with image gallery support!*


---


## ⚡ Quickstart
- Create the `output` folder and the model cache directory.
- All models are stored under `MODEL_BASE_PATH` (default `models`) and downloaded automatically from Hugging Face.
```bash
git clone https://github.com/arminfabritzek/WatchMyBirds.git
cd WatchMyBirds
cp docker-compose.example.yml docker-compose.yml
docker-compose up -d
```

---


## 📚 Table of Contents
- [Overview](#overview)
- [🚀 Key Features](#-key-features)
- [📡 Tested Cameras](#-tested-cameras)
- [📌 Share Your Results](#-share-your-results)
- [🌟 Roadmap](#-roadmap)
- [⚙️ Installation and Setup](#-installation-and-setup)
- [📺 Usage](#-usage)
- [🤝 Contributing](#-contributing)
- [🙌 Acknowledgements](#-acknowledgements)
- [📄 License](#-license)

---

## Overview

**WatchMyBirds** is a lightweight, customizable object detection application for real-time monitoring using webcams, RTSP streams, and Docker. It is built using PyTorch and TensorFlow, and it supports live video streaming, automatic frame saving based on detection criteria, and integration with Telegram for notifications. The application is ideal for hobbyists, researchers, and wildlife enthusiasts interested in automated visual monitoring.


---



## 🚀 Key Features

- **Real-Time Object Detection**:  
  Transform any webcam or RTSP camera stream into a powerful detection system.
  - Seamless integration with **MotionEye OS** for network camera support.  
  - Tested with cheap **IP** & **PTZ** Cameras.  


- **Optimized for Diverse Hardware**:  
  Built to run across various devices with performance and accuracy in mind.
  - Runs on Docker (e.g., Synology NAS), macOS, with planned support for Raspberry Pi and NVIDIA Jetson  


- **Integrated Notifications**  
  - Telegram alerts for detections  


- **State-of-the-Art AI Models**
  - Pre-trained models including `yolov11`, `EfficientNet`

- **Modular Capture Pipeline**
  - Explicit start/stop control via `VideoCapture` and timestamped streaming through `FrameGenerator`
- **Robust FFmpeg Lifecycle**
  - PID-tracked start/stop with parent-death signal + shutdown hooks to prevent orphaned FFmpeg processes after crashes or restarts
- **Stream Settings Cache**
  - First successful stream probe is persisted (`output/stream_settings.json` with validation: URL, stream type, FFmpeg version); cache writes are atomic and reused at restart to skip probing
- **Fast Web UI Startup**
  - Web UI starts immediately; stream status loads asynchronously with placeholders/fallbacks when the camera is not yet ready
- **Landing Page Context (Lightweight)**
  - Shows today's count, hourly chart, latest detections, and a lazy-loaded species summary without delaying initial render
  - Now also lazy-loads the gallery-style Daily Summary for today after first paint


---

## ⚙️ Configuration Highlights
- `VIDEO_SOURCE` is read at startup (no runtime change in this phase); UI still starts even if the stream is unavailable.
- `STREAM_FPS_CAPTURE` (default `0`) throttles frame capture; `0` disables throttling for freshest frames.
- `STREAM_FPS` (default `0`) throttles the UI MJPEG feed only; set a positive value to reduce UI bandwidth/CPU.
- Configuration is loaded once via `get_config()` and shared across modules to avoid divergent settings.
- `CPU_LIMIT` is applied via `restrict_to_cpus` using the shared config; set to 0 or below to skip affinity.
- Persistence: detections are stored in `OUTPUT_DIR/images.db` (SQLite, WAL) with the same fields as the former per-day CSV logs; rows now include `detector_model_id` / `classifier_model_id` for provenance.

---

## ⚙️ Configuration Reference (.env / docker-compose)
Set these as environment variables in your `.env` or docker-compose `environment:`. Defaults are shown.

| Setting | Default | Description |
| --- | --- | --- |
| `DEBUG_MODE` | `False` | Enable verbose logging and debug behavior. |
| `OUTPUT_DIR` | `/output` | Base directory for images and `images.db`. |
| `VIDEO_SOURCE` | `0` | Camera source (int for webcam, string for RTSP/HTTP). |
| `LOCATION_DATA` | `52.516, 13.377` | GPS lat/lon for EXIF (`"lat, lon"`). |
| `DETECTOR_MODEL_CHOICE` | `yolo` | Detection model selector (currently `yolo`). |
| `CONFIDENCE_THRESHOLD_DETECTION` | `0.55` | Detector confidence threshold for display logic. |
| `SAVE_THRESHOLD` | `0.55` | Detector confidence threshold to save a detection. |
| `MAX_FPS_DETECTION` | `0.5` | Target detection loop rate (FPS). |
| `MODEL_BASE_PATH` | `/models` | Base directory for model files. |
| `CLASSIFIER_CONFIDENCE_THRESHOLD` | `0.55` | Classifier confidence threshold for gallery summaries. |
| `FUSION_ALPHA` | `0.5` | Detector/classifier fusion weight in UI summaries. |
| `STREAM_FPS_CAPTURE` | `0` | Capture throttle (0 disables throttling). |
| `STREAM_FPS` | `0` | UI MJPEG feed throttle (0 disables throttling). |
| `STREAM_WIDTH_OUTPUT_RESIZE` | `640` | Width for the live stream preview in the UI. |
| `DAY_AND_NIGHT_CAPTURE` | `True` | Enable daylight gating for detections. |
| `DAY_AND_NIGHT_CAPTURE_LOCATION` | `Berlin` | City name for Astral daylight check. |
| `CPU_LIMIT` | `1` | CPU affinity cap (<=0 disables affinity). |
| `TELEGRAM_COOLDOWN` | `5` | Cooldown (seconds) between Telegram alerts. |
| `TELEGRAM_ENABLED` | `True` | Enables/disables Telegram sends (tokens remain env-only). |
| `EDIT_PASSWORD` | `SECRET_PASSWORD` | Password for edit page access in the UI. |

Telegram env vars (read directly by `utils/telegram_notifier.py`, not via `config.py`):

| Setting | Default | Description |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | (empty) | Bot token for notifications. |
| `TELEGRAM_CHAT_ID` | (empty) | Chat ID or JSON array of chat IDs. |

Unused settings: none found in current code; all keys in `config.py` are referenced.

---

## 🧩 Configuration Model (Boot vs Runtime)
The app uses two layers of configuration:

**Boot / Infrastructure (read-only at runtime)**  
Loaded once from `.env`/Docker and shown as read-only in the UI. Changes require a restart.

**Runtime Settings (UI-editable)**  
Stored in `OUTPUT_DIR/settings.yaml`, applied without changing startup semantics.

Merge order:
```
defaults → env → settings.yaml
```

Runtime edits update `settings.yaml` only (no `.env` mutation).

---

## 📡 Tested Cameras
| Camera Model                                   | Connection          | Status  | Notes                                                           |
|------------------------------------------------|---------------------|---------|-----------------------------------------------------------------|
| **Low-priced PTZ Camera**                      | RTSP                | ✅ Works | Stable RTSP stream verified.                                    |
| **Raspberry Pi 3 + Zero 2 + Raspberry Pi Cam** | MotionEye OS (HTTP Stream) | ✅ Works |                                                                 |

🔹 *Planned: Expanding RTSP camera compatibility & adding PTZ control.*

📢 *Interested in sponsoring a test? Reach out on GitHub!*

---


## 📌 Share Your Results
Your contributions help improve **WatchMyBirds** for everyone! 🚀



---

## 🌟 Roadmap

### 🧠 AI & Model Optimization
- 🏆 Train custom bird/insect/plant models with classifiers  

### ⚡ Performance & Edge Deployment
- 🏆 Optimize for Raspberry Pi 4/5 and Jetson Nano  

### 📊 Analytics & Visualization
- 🏆 Track bird visits, diversity, and time patterns  
- 🏆 Interactive dashboards for visualization  

---


## ⚙️ Installation and Setup

---
### Using Docker


1. Clone the repo and copy the example compose file:
   ```bash
   cp docker-compose.example.yml docker-compose.yml
   ```

2. Edit the .yml file to match your stream settings, then run:

    ```bash
    docker-compose up -d
   ```

➕ See [`docker-compose.example.yml`](docker-compose.example.yml) for all available environment variables.


This will run the **WatchMyBirds** application, and you can access the livestream at `http://<your-server-ip>:8050`.
The image creates `/models` and `/output` at build time; models are downloaded at runtime as needed.



---
### Manual Setup (Without Docker)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arminfabritzek/WatchMyBirds.git
   cd WatchMyBirds
   ```

2. **Set Up a Virtual Environment** (optional but recommended; tested on Python 3.10 and 3.12):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` lists direct dependencies only; pip will resolve transitive packages.


4. **Configure the Video Source**:
- Create or edit the .env file in the project root

For a webcam connected via USB use:
   ```plaintext
   VIDEO_SOURCE=0
   ```

   For an RTSP stream, use:
   ```plaintext
   VIDEO_SOURCE=rtsp://user:password@192.168.0.2:554/1
   VIDEO_SOURCE=rtsp://user:password@192.168.0.2:8554/stream1
   ```

5. **Start the Application**:
   ```bash
   python main.py
   ```

Windows notes:
- Gallery image URLs are generated with forward slashes for compatibility. If you previously saw broken thumbnails, hard-refresh the page after updating.


## 📺 Usage
   The livestream will be available at:
   - Local: `http://localhost:8050`
   - Remote: `http://<your-server-ip>:8050`

---

## 🤝 Contributing

Have ideas or improvements? Open an issue or submit a pull request!


---

## 🙌 Acknowledgements

This project uses **Label Studio** – provided free through the Academic Program by HumanSignal, Inc.  
[![Label Studio Logo](https://user-images.githubusercontent.com/12534576/192582340-4c9e4401-1fe6-4dbb-95bb-fdbba5493f61.png)](https://labelstud.io)

---

## 📄 License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
