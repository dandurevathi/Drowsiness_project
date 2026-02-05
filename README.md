#Vision-Based Driver Drowsiness & Distraction Detection System
 Real-Time Monitoring using Transfer Learning and Edge-Optimized Inference

🔍 Motivation & Core Idea
Driver drowsiness and distraction are among the leading causes of road accidents, particularly during long-duration and night-time driving. Many existing solutions rely on intrusive sensors or expensive hardware, limiting their practicality for large-scale deployment.
The core idea of this project is to design a vision-based driver monitoring system that leverages computer vision and deep learning to detect driver alertness in real time, while being lightweight enough to run on edge devices such as Raspberry Pi.
This project emphasizes practical deployment and system-level thinking, rather than model accuracy alone.

🎯 Project Objective
Monitor driver alertness in real time using facial cues
Detect and classify driver states as Active, Distracted, or Drowsy
Utilize transfer learning for efficient model training
Optimize inference for edge-device deployment
Integrate visual and audio feedback mechanisms for timely alerts

✨ Key Features
Real-time facial landmark detection using MediaPipe
Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)–based behavioral analysis
Lightweight CNN based on MobileNetV2
Pretrained and optimized TensorFlow Lite (TFLite) model for fast inference
Audio alert system to warn the driver during drowsy states
Modular architecture enabling easy future enhancements

🧠 Model & Technical Approach
Model Backbone: MobileNetV2 (pretrained on ImageNet)
Learning Strategy: Transfer Learning
Utilizes pretrained feature extraction for edges, textures, and facial patterns
Reduces training time and improves generalization on limited data
Optimization:
Trained Keras model converted to TFLite
Designed for low-latency inference on edge devices
Decision Logic:
CNN-based classification combined with EAR/MAR thresholds for robust detection

🛠️ Technologies Used
Programming & Frameworks
Python 3.10
TensorFlow / Keras
TensorFlow Lite
Computer Vision & ML
OpenCV
MediaPipe
Transfer Learning (MobileNetV2)
Hardware & Deployment
Laptop Webcam (live demo)
Raspberry Pi 5 (edge deployment and testing completed)

Running the Project
1️⃣ Clone the repository
2️⃣ Install dependencies 
pip install opencv-python mediapipe tensorflow numpy
3️⃣ Run the real-time detection
python fusion_code.py
The system will activate the webcam and display the driver’s current state in real time.

⛔ How to Stop Execution
press Q in the OpenCV display window (if enabled)

🧪 Edge Device Deployment (Completed)
The system has been successfully integrated and tested on Raspberry Pi 5, validating its suitability for edge-based deployment. The TFLite model enables on-device inference without the need for retraining.
During testing, a minor inference latency was observed due to real-time video capture, facial landmark extraction, and CPU-based model execution. This reflects realistic edge-computing constraints and provides valuable insight for further optimization.

🎓 Key Learnings
End-to-end development of a real-time vision-based ML system
Practical application of transfer learning in constrained environments
Model optimization and conversion for edge devices using TFLite
Understanding performance trade-offs in real-world deployments
Integrating deep learning with classical computer vision techniques

🚀 Future Enhancements
Model quantization and frame-skipping to further reduce latency
Multi-threaded inference pipeline for improved throughput
Adaptive driver-specific alert thresholds
Integration with vehicle systems for automated safety responses

🎥 Demonstration
Laptop: Live webcam-based real-time demo
Edge Device: Raspberry Pi 5 deployment and validation completed

Final Note
This project includes a pretrained, edge-optimized model, enabling immediate real-time inference without retraining. The focus is on practical applicability, system integration, and deployment readiness, making it suitable for real-world intelligent transportation systems.
