# AIC-GM-Comp2025
# 🛡️ A Privacy-Preserving Approach for Safe Navigation 🔭
 
Welcome to the official repository for **Group 32's winning submission** to Challenge 3: *Privacy in Data*, presented at the GM Hackathon! This project explores efficient and accurate road user detection using **thermal imagery**—all while preserving user privacy.

Get ready for a wild ride—with no need for a driver's license, just your curiosity.

---

## 🚗 Overview

As the automotive industry continues to rely heavily on real-time data, two priorities emerge: **efficiency** and **accuracy**. Our task was to develop a lightweight, fast, and privacy-preserving object detection system optimized for **Raspberry Pi 5** using thermal images.

---

## 🧠 Why YOLO?

Although we explored options like **SAM** and **OpenCV-based custom models**, **YOLO (You Only Look Once)** emerged as the ideal choice due to:

- ⚡ **Lightning-Fast Inference**: Real-time performance on edge devices like the Raspberry Pi.
- 🔧 **Streamlined Development Pipeline**: From training to quantization and pruning.
- 🌍 **Field-Tested Versatility**: Robust, well-supported, and widely adopted.

---

## 🔧 Optimization Strategies

### 🧮 Model Efficiency (PyTorch)

- **Quantization (FP16)**: Reduced activation precision from 32-bit to 16-bit.  
  ⮕ *~1% accuracy drop, but a 1.2× speedup.*
  
- **Pruning**: Removed 20% of the least significant neurons.  
  ⮕ *Trained for 50 epochs, batch size 16. Minimal accuracy loss.*

- **ONNX + NCNN**: Deployed using [NCNN](https://github.com/Tencent/ncnn) for ARM—an efficient C++ inference framework.  
  ⮕ *Perfect for Raspberry Pi with seamless ONNX integration.*

---

## 🍓 Raspberry Pi 5 Optimizations

- 🔁 **OS Upgrade**: Switched from full PiOS to **headless 64-bit PiOS**.  
  ⮕ *~20% efficiency gain.*

- ⚙️ **C++ Scripting**: Replaced Python to reduce latency.

- 🚀 **Overclocked Pi**: Boosted from **2.4GHz → 2.7GHz**  
  ⮕ *Performance increased by 12%.*

- 🧠 **Memory Reallocation**: GPU limited to **16MB**, freeing **8GB RAM** for CPU.  
  ⮕ *Full model ran on CPU for max efficiency.*

---

## 🧪 Model Tuning & Results

| Metric               | Before       | After        |
|---------------------|--------------|--------------|
| Accuracy            | 71%          | **81%**      |
| Runtime (seconds)   | 1.13         | **0.625**    |
| Time Saved          | —            | **0.505 s**  |

- **Model variations**: Nano, Small, Medium  
- **Hyperparameters**: Epochs, Batch Size, Train:Val Split  
- **Training Framework**: **YOLO + PyTorch**

---

## 🔐 Privacy First: Federated Learning

To further enhance user privacy, we incorporated **federated learning**:

> A centralized model is distributed to local clients who train on real-world data and return only updated weights.  
> **No raw data is shared**, protecting sensitive info like driver identity, location, and vehicle type.

Inspired by a keynote on federated learning, we built this with future scalability and automotive-grade security in mind.

---

## 👥 Team 32

- **Mohammed Owda** – Software Engineering  
- **Ethan Zhao** – Software Engineering  
- **Aly Soliman** – Software Engineering  
- **Benjamin Namayandeh** – Mechatronics Engineering  

---

## 🏁 Conclusion

Through smart model selection, precise tuning, hardware optimization, and privacy-focused design, our solution tackled real-world constraints while delivering impressive speed and accuracy.

Thank you for checking out our work—**buckle up for the next adventure in innovation!** 🚀

---

