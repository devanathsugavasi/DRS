# Cricket DRS - Project Overview

This is an **engineering-grade prototype** that mimics the highly sophisticated, multi-million dollar broadcast technology used in international cricket (like Hawk-Eye, UltraEdge, and HotSpot). It includes strict **Accuracy Gates**—meaning it refuses to give an "OUT" or "NOT OUT" decision unless the system meets professional-level confidence thresholds (like sub-8 millisecond sync error and high camera calibration accuracy)—showing a mature, production-ready engineering mindset. It avoids guessing when the data is poor, safely falling back to `REVIEW INCONCLUSIVE`.

## What is this project exactly?
It is a complete **Cricket Decision Review System** pipeline. It takes raw video feeds from multiple cameras (up to 6) around a cricket pitch, tracks the cricket ball in real-time, and provides umpires and broadcasters with the analytical tools needed to make decisions on LBW (Leg Before Wicket) and catches (edges).

The project is broken down into several modular pieces:
1. **The AI Vision Core**: Uses YOLO (versions 8 or 11) to detect the ball.
2. **The Backend Engine**: A Python/FastAPI server that crunches the math, physics, and synchronization.
3. **The User Interfaces**: 
   - An **Electron app** meant to act as the "Broadcast Command Center" (what the TV operators would use).
   - A **React Testing Platform** for developers to upload videos and test the system locally.

## How it Works (The Workflow)

Tracing a single cricket delivery through the system:

1. **Multi-Camera Syncing**: The system ingests video from 2 to 6 cameras. Because the ball moves incredibly fast, the `Sync manager` ensures that frames from different cameras are perfectly aligned in time (within 8 milliseconds).
2. **Ball Detection**: The synchronized frames are passed to a **YOLO model** (like `yolo11l.pt`). The model's only job is to find the cricket ball in every single 2D frame.
3. **Tracking & Physics (ByteTrack + EKF)**: Once the ball is found in individual frames, the system uses a tracker to "connect the dots" frame-by-frame. It uses an Extended Kalman Filter (EKF) to smooth out the trajectory and predict where the ball is going, even if a frame drops.
4. **3D Reconstruction & Calibration**: The system uses checkerboard calibration data (reprojection errors, homography) to translate the 2D video coordinates into a real-world 3D map of the cricket pitch. 
5. **The Analysis Engines**:
   - **LBW Gates**: Calculates the 3D trajectory to determine where the ball pitched, where it impacted the batter, and if it was hitting the stumps.
   - **UltraEdge / Audio Analyzer**: Analyzes the audio track synced with the video to look for microscopic sound spikes that indicate the ball hit the bat.
   - **HotSpot Simulation**: Simulates the infrared edge-detection technology.
6. **Decision Rendering**: All of this data is bundled up by the FastAPI backend and sent via WebSockets to the frontend dashboards. 
7. **The Final Gate**: Before showing the result on the screen, the system checks its **Accuracy Gates**. If the AI is at least 88% confident, the calibration is off by less than 5cm, and the frame rate is good, it will confidently display `OUT` or `NOT OUT`. Otherwise, it plays it safe and displays `REVIEW INCONCLUSIVE`.
