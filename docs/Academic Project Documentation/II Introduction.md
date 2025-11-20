INTRODUCTION

The development of autonomous vehicles represents one of the most significant technological challenges of the 21st century. Central to this challenge is the ability of vehicles to perceive and interpret their environment accurately and in real-time. Among the critical components of autonomous driving systems is the capability to recognize and classify traffic signs, which provide essential regulatory and warning information for safe navigation. Traffic sign recognition systems must achieve exceptionally high accuracy rates, typically above 95%, as errors in this domain can have serious safety implications for passengers, pedestrians, and other road users.

Traditional computer vision approaches to traffic sign recognition relied heavily on hand-crafted features and classical machine learning algorithms. These methods often struggled with variations in lighting conditions, weather, viewing angles, partial occlusions, and sign degradation. The emergence of deep learning, particularly convolutional neural networks, has revolutionized this field by enabling systems to automatically learn hierarchical feature representations directly from raw image data. Transfer learning, which leverages pre-trained models on large-scale datasets, has further accelerated development by providing robust feature extractors that can be fine-tuned for specific tasks.

This project addresses the traffic sign recognition problem as a foundational component of a comprehensive self-driving car system. By developing a highly accurate classifier, we aim to create a reliable module that can be integrated into a multi-model autonomous driving architecture alongside pedestrian detection, lane tracking, and vehicle classification systems.


A. Problem Statement

The primary challenge addressed in this project is the development of an automated traffic sign recognition system capable of achieving state-of-the-art performance on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Specific problems include:

1. Classification Accuracy Requirements
   The safety-critical nature of autonomous driving demands classification accuracy exceeding 95%. False negatives (failing to detect a stop sign) or false positives (misclassifying a speed limit) can lead to dangerous driving behaviors.

2. Class Imbalance in Real-World Data
   Traffic sign datasets exhibit severe class imbalance, with some sign types appearing 10-40 times more frequently than others. This imbalance can cause models to develop biases toward majority classes, resulting in poor performance on rare but critical signs.

3. Environmental Variability
   Traffic signs must be recognized under diverse conditions including varying lighting (day, night, shadows), weather (rain, fog, snow), viewing angles, distances, and states of degradation or partial occlusion.

4. Real-Time Processing Constraints
   For practical deployment in self-driving vehicles, the recognition system must process images rapidly enough to enable timely decision-making, typically requiring inference times under 100 milliseconds.

5. Generalization Across Different Contexts
   A robust system must generalize from close-up training images to real-world scenarios where signs may appear at various distances and within complex visual scenes containing multiple objects.

The ultimate goal of this project is to develop a traffic sign classifier that serves as a foundational component for a future self-driving car system, where multiple specialized models work in concert to enable autonomous navigation.


B. Proposed Solution

This project implements a deep learning-based solution using the ResNet50 architecture with transfer learning to address the traffic sign classification challenge. The key components of the proposed solution include:

1. Deep Residual Network Architecture (ResNet50)
   ResNet50, a 50-layer deep convolutional neural network with residual connections, serves as the backbone of our classifier. This architecture addresses the vanishing gradient problem through skip connections, enabling effective training of very deep networks. The model contains 23.5 million parameters and has demonstrated excellent performance across numerous computer vision tasks.

2. Transfer Learning from ImageNet
   Rather than training from scratch, we leverage a ResNet50 model pre-trained on ImageNet, a dataset containing 1.2 million images across 1,000 categories. This pre-training provides robust low-level and mid-level feature extractors (edge detectors, texture analyzers, shape recognizers) that are highly relevant to traffic sign recognition. We modify only the final classification layer to output 43 classes corresponding to GTSRB traffic signs.

3. Comprehensive Data Preprocessing and Augmentation
   To improve model robustness and generalization, we implement extensive data augmentation including random rotations, affine transformations, color jittering, perspective distortions, and random erasing. Region of Interest (ROI) cropping focuses the model's attention on the actual sign content by removing background noise.

4. Advanced Training Strategies
   The training process incorporates multiple regularization techniques to prevent overfitting: dropout layers (40%), weight decay (L2 regularization), gradient clipping, and early stopping. To address class imbalance, we employ weighted random sampling and weighted cross-entropy loss, ensuring the model learns to recognize rare signs effectively.

5. Robust Evaluation and Model Selection
   We utilize a three-way data split (training, validation, and test sets) to ensure unbiased performance evaluation. The best model is selected based on validation accuracy, and final performance is assessed on a completely held-out test set from the official GTSRB benchmark.

This solution directly addresses the identified problems by:
- Achieving target accuracy through deep learning and transfer learning
- Handling class imbalance via weighted sampling and loss functions
- Increasing robustness through extensive data augmentation
- Demonstrating generalization through strong test set performance
- Providing a modular component suitable for integration into a multi-model autonomous driving system

The trained model achieves 100% validation accuracy and 99.11% test accuracy, demonstrating that this approach effectively solves the traffic sign classification problem and provides a solid foundation for future autonomous vehicle development.