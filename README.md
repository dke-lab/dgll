![image](https://github.com/user-attachments/assets/b5e21e90-4d70-4495-b5b5-690d15fa45ca)

# Distibuted Graph Learning Library
Welcome to the Distributed Graph Learning Library.

Distibuted Graph Learning Library (DGLL) is a high-performance distributed graph learning framework built to scale Graph Neural Network (GNN) training to massive graphs across multi-node, multi-GPU environments. It combines optimized communication patterns with advanced memory and compute strategies to address the key bottlenecks in distributed GNN workloads. The library includes support for intelligent caching to minimize redundant data movement, GPU-friendly binarization techniques to accelerate matrix operations, and custom implementations of Sparse Matrix-Matrix Multiplication (SPMM) and Dense Matrix-Matrix Multiplication (Dense MM) tailored for graph data. With flexible graph partitioning, pluggable deep learning backends, and a modular architecture, this library serves as a powerful foundation for developing and deploying large-scale GNN models in both research and production environments.


## Features
- **Efficient Caching**: Reduce network overhead with intelligent, distributed caching mechanisms.
- **Binary Operations**: Speed up GPU compute using binarized representations for key matrix operations.
- **Custom Sparse and Dense Matrix Multiplication**:
  - Optimized **SPMM** tailored for graph structures.
  - High-throughput **Dense MM** operations for deep GNN layers.
- **Scalable Distributed Training**: Train GNNs on large-scale graphs across multiple machines with minimal communication cost.
- **Pluggable Backend**: Seamlessly integrates with popular deep learning frameworks.
- **Flexible Graph Partitioning**: Supports various partitioning strategies with minimal data duplication.


## Architecture
Our distributed GNN training framework's modular design is organized into three main layers. At the top, the Graph Machine Learning layer provides high-level components such as GNN models, sampling, transformations, and tensor operations, enabling users to define and manipulate graph-based deep learning workflows. The Distributed Training Launcher layer coordinates the execution of distributed jobs, handling multithreaded task scheduling, inter-process communication, and orchestration of servers and trainers. At the base, the Distributed Engine integrates with PyTorch Distributed and ZeroMQ, supporting scalable data parallelism and robust message-passing. Core components like DistGraph and Graph Services manage partitioned graph storage, efficient neighbor access, and backend server functionalities.

![image](https://github.com/user-attachments/assets/3b32180c-a64e-4d0a-b3d0-45f213db8709)


## MQ GNN
MQ-GNN, a multi-queue pipelined framework that maximizes training efficiency by interleaving GNN training stages and optimizing resource utilization. MQ-GNN introduces Ready-to-Update Asynchronous Consistent Model (RaCoM), which enables asynchronous gradient sharing and model updates while ensuring global consistency through adaptive periodic synchronization. Additionally, it employs global neighbor sampling with caching to reduce data transfer overhead and an adaptive queue-sizing strategy to balance computation and memory efficiency.

The Following figure illustrates the MQ-GNN’s seven-stage pipelined architecture, showing data flow across CPU and GPU for tasks like mini-batch generation, computation, and model updates. It depicts three queues (CPU/GPU mini-batch, gradient) enabling concurrent operations, with Global Neighbor Sampling caching nodes in GPU memory. The diagram highlights asynchronous gradient sharing via RaCoM, with synchronization points ensuring model consistency, emphasizing scalability and efficiency.

![MQ GNN Arechitecture](https://github.com/user-attachments/assets/a37f83f9-dc0e-47d4-95c7-95340b93dda5)


### RaCoM
The following figure focuses on RaCoM, showing a timeline of gradient enqueuing, asynchronous sharing, and model updates across GPUs. Gradient queues on each GPU are depicted, with arrows indicating non-blocking exchanges. Periodic synchronization, based on graph size and GPU count, is marked, illustrating how MQ-GNN minimizes communication overhead while maintaining model accuracy.

![RACoM_Fig](https://github.com/user-attachments/assets/18b82b22-3f87-4b3b-8b94-318e037c400b)

### GPU Utilization
The following figure compares GPU utilization during a single epoch of training on the Reddit dataset using MQ-GCN and GCN. It shows MQ-GNN achieving a 1.98× speedup and 30% higher GPU utilization, averaging 64.2% with a peak of 73%, compared to DGL’s 39.42%. The figure likely illustrates optimal queue sizing and interleaved data transfer and computation, minimizing GPU starvation, as depicted in a bar chart or time-series graph.

![image](https://github.com/user-attachments/assets/8af25da9-17c8-4f60-b710-ee1ecbcebb08)

### Performance
The following table lists performance metrics for various models and their variants on a single GPU. MQ-FastGCN+f+d achieves the maximum 2.06× training time speedup on ogbn-products (784,496.26 ms to 378,999.76 ms), with batch time reduced from 20.22 ms to 6.12 ms. Accuracy remains consistent, showcasing MQ-GNN’s efficiency via pipelining and caching, which minimizes redundant node access and GPU starvation on large datasets.

![table4](https://github.com/user-attachments/assets/e1ad910f-3fa4-4edc-beac-497b41bc055b)

The following table shows performance metrics for LADIES variants on four GPUs. MQ-LADIES+f+d achieves the highest 4.6× training time speedup on Reddit (247,618.18 ms to 56,834.54 ms), with batch time dropping from 11.00 ms to 2.44 ms.

![table13](https://github.com/user-attachments/assets/e9785a3e-69b1-4962-81a8-422aea93077e)

___
If you use this work, please cite the following publications:
- Ullah, Irfan, and Young-Koo Lee. "MQ-GNN: A Multi-Queue Pipelined Architecture for Scalable and Efficient GNN Training." IEEE Access (2025).
- Van, Duong Thi Thu, et al. "GDLL: a scalable and share nothing architecture based distributed graph neural networks framework." IEEE Access 10 (2022): 21684-21700.
- Morshed, Md Golam, Tangina Sultana, and Young-Koo Lee. "FaLa: feature-augmented location-aware transformer network for graph representation learning." Neural Computing and Applications (2025): 1-28.
- Morshed, Md Golam, Tangina Sultana, and Young-Koo Lee. "LeL-GNN: Learnable edge sampling and line based graph neural network for link prediction." IEEE Access 11 (2023): 56083-56097.
- Morshed, Md Golam, and Young-Koo Lee. "LFP: Layer Wise Feature Perturbation based Graph Neural Network for Link Prediction." 2023 IEEE International Conference on Big Data and Smart Computing (BigComp). IEEE, 2023.
