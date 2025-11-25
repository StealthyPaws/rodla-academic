# Federated RoDLA: A Comprehensive Guide
**Benchmarking the Robustness of Document Layout Analysis Models in a Federated Setting**

---

## 1. Introduction

### 1.1 What is RoDLA?
**RoDLA (Robust Document Layout Analysis)** is a benchmark designed to evaluate how well Document Layout Analysis (DLA) models perform when faced with real-world degradations. Standard DLA models (like Faster R-CNN or Mask R-CNN trained on PubLayNet) often fail when documents are scanned imperfectly, photographed with poor lighting, or contain watermarks.

The RoDLA benchmark introduces **12 distinct types of perturbations** (e.g., motion blur, ink bleeding, paper texture) across **3 severity levels** to rigorously test model robustness.

### 1.2 Why Federated Learning?
In real-world scenarios, document datasets (financial reports, medical records, legal contracts) are highly sensitive and private. Organizations cannot upload this data to a central server for training due to GDPR, HIPAA, or internal privacy policies.

**Federated Learning (FL)** solves this by reversing the standard training paradigm:
*   **Standard Training**: Move data to the model (Central Server).
*   **Federated Training**: Move the model to the data (Clients).

In **Federated RoDLA**, we combine these two concepts. We train a DLA model across multiple decentralized clients (simulating different hospitals or banks) without ever sharing their raw documents. We then evaluate the resulting global model using the RoDLA robustness benchmark to ensure that privacy preservation does not come at the cost of model fragility.

---

## 2. System Architecture

The system is built upon the **Client-Server** model using **HTTP/REST** for communication.

### 2.1 The Central Server (`start_federated_server.py`)
The server is the "conductor" of the orchestra. It does not perform training and does not see data.
*   **Role**: Aggregator & Coordinator.
*   **State**: Maintains the current `Global Model` weights.
*   **Algorithm**: Implements **FedAvg (Federated Averaging)**.
    *   Mathematically: $W_{global} = \frac{1}{N} \sum_{i=1}^{N} W_{client_i}$
*   **Endpoints**:
    *   `POST /register_client`: Adds a client to the active pool.
    *   `GET /get_global_model`: Distributes current weights to clients.
    *   `POST /submit_model_update`: Receives trained weights from clients.

### 2.2 The Federated Client (`start_federated_client.py`)
The client is the "musician". It possesses the talent (compute) and the sheet music (private data).
*   **Role**: Local Trainer.
*   **Engine**: Uses **MMDetection** (PyTorch) for the heavy lifting.
*   **Process**:
    1.  **Pull**: Downloads global weights.
    2.  **Train**: Fine-tunes the model on local PubLayNet data for $E$ epochs.
    3.  **Push**: Uploads the *difference* (or new weights) to the server.
*   **Privacy**: Raw images (`.jpg`) and annotations (`.json`) never leave the local disk.

### 2.3 The Robustness Evaluator (`run_robustness_evaluation.py`)
This is the "critic". It runs *after* the concert (training) is finished.
*   **Role**: Benchmarking Tool.
*   **Method**: It takes the final Global Model and runs it against a clean test set that is dynamically corrupted on-the-fly using the **RoDLA Perturbation Engine**.
*   **Metric**: **mAP (mean Average Precision)** @ IoU 0.50:0.95.

---

## 3. The RoDLA Perturbation Engine
**File**: `federated/perturbation_engine.py`

This engine is the heart of the robustness benchmark. It implements the 12 perturbations defined in the RoDLA paper. These are applied **only during evaluation**, simulating "Out-of-Distribution" (OOD) testing.

| Category | Perturbation Type | Description | Real-world Analogy |
| :--- | :--- | :--- | :--- |
| **Camera** | `defocus` | Gaussian blur application | Out-of-focus camera lens |
| | `motion_blur` | Linear motion blur (simulated via `vibration`) | Shaking hand while scanning |
| **Lighting** | `illumination` | Brightness/Contrast shifts | Poor scanner settings / shadows |
| | `speckle` | Additive Gaussian noise | ISO noise in low light |
| **Paper** | `texture` | Blending paper texture overlays | Recycled or rough paper |
| | `ink_bleeding` | Morphological dilation | Low-quality ink/paper absorption |
| | `ink_holdout` | Random pixel dropout | Faded toner / printer skip |
| **Geometry** | `rotation` | Affine rotation | Crooked scanning |
| | `keystoning` | Perspective transform | Taking photo at an angle |
| | `warping` | Sinusoidal mesh warping | Crumpled or bent paper |
| **Content** | `watermark` | Alpha-blended text overlay | "CONFIDENTIAL" stamps |
| | `background` | Color shifting | Colored paper stock |

**Severity Levels**:
Each perturbation has 3 levels ($s=1, 2, 3$).
*   **Level 1**: Barely visible (e.g., 2° rotation).
*   **Level 2**: Noticeable (e.g., 5° rotation).
*   **Level 3**: Severe degradation (e.g., 10° rotation, heavy noise).

---

## 4. Detailed Implementation Walkthrough

### 4.1 Server Logic (`start_federated_server.py`)
The server uses `Flask` to handle concurrent requests.
*   **`_perform_aggregation()`**: This is the critical function.
    ```python
    # Simplified Logic
    for key in global_weights:
        global_weights[key] = sum(client_weights[key] for client in clients) / num_clients
    ```
    It iterates through the state dictionaries of all submitted models, sums the tensors layer-by-layer, and divides by the count. This new average becomes the starting point for the next round.

### 4.2 Client Logic (`start_federated_client.py`)
The client wraps MMDetection's `train_detector` API.
*   **`_train_local_model()`**:
    1.  **Config Loading**: Reads `configs/publaynet/rodla_config.py`.
    2.  **Weight Injection**: Overwrites the model's initial weights with the base64-decoded weights received from the server.
    3.  **Training**: Runs a standard PyTorch training loop (Forward -> Loss -> Backward -> Optimizer Step) for `local_epochs`.
    4.  **Extraction**: Extracts the new `state_dict` from the GPU/CPU.
    5.  **Serialization**: Serializes the `state_dict` to a BytesIO buffer -> Base64 string -> JSON payload.

### 4.3 Evaluation Logic (`run_robustness_evaluation.py`)
*   **Dynamic Corruption**: Unlike standard datasets where you might have `test_clean` and `test_noisy` folders, this script generates noisy images in RAM.
    *   It loads a clean image.
    *   It calls `PubLayNetPerturbationEngine.perturb(image, type='defocus', severity=2)`.
    *   It feeds the result to the model.
    *   It compares the prediction to the *original* clean ground truth (since layout doesn't change with noise, only visibility does).

---

## 5. Deployment & Usage Guide

### 5.1 Environment Setup
**Prerequisites**:
*   Anaconda or Miniconda
*   NVIDIA GPU (Recommended) or CPU (Slow but functional)

```bash
# 1. Create Environment
conda create -n RoDLA python=3.8 -y
conda activate RoDLA

# 2. Install PyTorch (Check pytorch.org for your specific CUDA version)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 3. Install MMCV-Full (MUST match PyTorch/CUDA version exactly)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# 4. Install MMDetection & Utilities
pip install mmdet flask requests numpy pillow opencv-python
```

### 5.2 Data Preparation
You need the **PubLayNet** dataset.
*   **Images**: `train/` directory containing JPGs.
*   **Annotations**: `annotations/train.json` (COCO format).
*   *Note*: For FL, you can split this dataset across laptops. Laptop A gets the first 500 images, Laptop B gets the next 500, etc.

### 5.3 Execution: The 3-Laptop Setup
**Objective**: Run 1 Server and 2 Clients.

#### **Machine 1: The Server**
*   **IP**: `192.168.1.10` (Example)
*   **Command**:
    ```bash
    python scripts/start_federated_server.py --host 0.0.0.0 --port 8080 --rounds 10 --clients-per-round 2
    ```
*   **Output**: "Server started. Waiting for clients..."

#### **Machine 2: Client A**
*   **Data Location**: `D:/Datasets/PubLayNet/subset_A`
*   **Command**:
    ```bash
    python scripts/start_federated_client.py \
        --client-id client_A \
        --server-url http://192.168.1.10:8080 \
        --config configs/publaynet/rodla_config.py \
        --data-root D:/Datasets/PubLayNet/subset_A \
        --annotation-file D:/Datasets/PubLayNet/annotations/subset_A.json \
        --local-epochs 1
    ```

#### **Machine 3: Client B**
*   **Data Location**: `C:/Users/Data/PubLayNet/subset_B`
*   **Command**:
    ```bash
    python scripts/start_federated_client.py \
        --client-id client_B \
        --server-url http://192.168.1.10:8080 \
        --config configs/publaynet/rodla_config.py \
        --data-root C:/Users/Data/PubLayNet/subset_B \
        --annotation-file C:/Users/Data/PubLayNet/annotations/subset_B.json \
        --local-epochs 1
    ```

### 5.4 The "Magic" (What happens next)
1.  **Registration**: Both clients ping the server. Server sees "2/2 clients ready".
2.  **Round 1 Start**: Server unlocks the global model.
3.  **Download**: Clients download the ~300MB model file.
4.  **Local Training**: Clients start their GPUs. You will see MMDetection logs (`Epoch [1][10/50] lr: ... loss: ...`).
5.  **Upload**: Clients finish and POST their weights.
6.  **Aggregation**: Server computes the average.
7.  **Loop**: Process repeats for Round 2.

---

## 6. Interpreting Results

After training, you run the evaluation script.

```bash
python scripts/run_robustness_evaluation.py ...
```

**Sample Output**:
```text
Evaluating with Perturbation: defocus, Severity: 1
--> mAP: 0.852 (Clean baseline was 0.880)
Evaluating with Perturbation: defocus, Severity: 3
--> mAP: 0.620 (Significant drop!)
Evaluating with Perturbation: watermarks, Severity: 3
--> mAP: 0.810 (Model is robust to watermarks)
```

**Analysis**:
*   **High mAP on Severity 3**: The model is "Robust".
*   **Low mAP on Severity 1**: The model is "Fragile".
*   **Goal**: The goal of RoDLA is to maximize these numbers. In a Federated setting, we want to see if the *collaborative* model is more robust than a *single-client* model, or if the aggregation process introduces instability.

---

## 7. Troubleshooting

*   **Connection Refused**:
    *   Is the Server running?
    *   Is the IP correct?
    *   **Firewall**: Disable Windows Firewall on the Server or allow port 8080.
*   **OOM (Out of Memory)**:
    *   Edit `start_federated_client.py`: Change `cfg.data.samples_per_gpu = 2` to `1`.
*   **Dimension Mismatch**:
    *   Ensure all clients use the **exact same config file**. If Client A uses a ResNet-50 backbone and Client B uses ResNet-101, averaging will fail catastrophically.

---

## 8. Credits & References
*   **Original Paper**: *RoDLA: Benchmarking the Robustness of Document Layout Analysis Models* (Chen et al., 2024).
*   **Base Framework**: MMDetection (OpenMMLab).
*   **Dataset**: PubLayNet (IBM).
