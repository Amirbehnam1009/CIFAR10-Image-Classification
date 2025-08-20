# 🖼️ CIFAR-10 Image Classification Using Fully Connected Neural Network  
*"From foundational principles to optimized implementation - a complete neural network journey"* 🧠  

---

## 📋 Project Overview  
In this comprehensive project, we constructed a neural network from scratch using fundamental principles, leveraging the CIFAR-10 dataset for training. The implementation encompassed data preprocessing, forward/backward propagation, and highlighted the transformative power of vectorization for computational efficiency.  

**Core Insight:** This project demonstrates the pivotal role of vectorization in accelerating neural network computations, illuminating its critical importance in modern machine learning workflows. ⚡  

---

## 👨‍🏫 Supervision  
Under the guidance of **Prof. Mohammad Mehdi Ebadzadeh**  
📅 Spring 2022  

---

## 📚 Libraries Used  
- **NumPy** - Fundamental package for scientific computing  
- **Matplotlib** - Comprehensive library for visualization  
- **Scikit-image** - Image processing algorithms  
- **PIL (Pillow)** - Image manipulation capabilities  
- **Glob** - File path pattern matching  
- **OS** - Operating system interface  
- **Time** - Time access and measurement utilities  

---

## 🚀 Implementation Steps  

### 1. 🛠️ Data Preprocessing  
**Step 1: Data Acquisition & Storage**  
- 📥 Read the first 4 classes from CIFAR-10 dataset (airplane ✈️, automobile 🚗, bird 🐦, and cat 🐱)  
- 💾 Store data in matrix format: `(n_samples, width, height, channels)`  
- 🔢 Encode labels using one-hot representation  

**Steps 2–5: Data Transformation Pipeline**  
- 🎶 Grayscale Conversion - Reduce computational complexity by converting RGB to grayscale  
- 📊 Normalization - Scale pixel values to `[0, 1]` range by dividing by 255  
- 🧩 Flattening - Reshape data to `(n_samples, 1024)` for input layer compatibility  
- 🔀 Shuffling - Randomize data order while maintaining data-label correspondence  

---

### 2. 📈 Feedforward Implementation  
Objective: Compute network outputs using forward propagation  

- ✅ Data Selection: 200-sample subset from training data  
- ✅ Parameter Initialization:  
  - 🎲 Random weight initialization  
  - 0️⃣ Zero bias initialization  
- ✅ Output Computation: Matrix multiplication + Sigmoid activation σ  
- ✅ Model Inference: Class prediction based on maximum activation 📈  
- ✅ Accuracy Assessment: ~25% baseline accuracy (random chance) 🎯  

**Implementation Note:** Leveraged NumPy for efficient matrix operations  

---

### 3. 🔁 Backpropagation Implementation  
- Employed backpropagation to iteratively refine model parameters and minimize prediction error  
- ⚙️ Hyperparameter Tuning: Careful selection of batch size, learning rate, and epochs  
- 🔧 Algorithm Implementation: Followed standard pseudo-code for parameter updates  

**📊 Performance Metrics:**  
- Model accuracy on 200-sample subset  
- Execution time measurement ⏱️  
- Expected performance: ~30% accuracy (accounting for random initialization)  
- 📉 Cost Visualization: Plotted average cost reduction per epoch  

---

### 4. ⚡ Vectorization Optimization  
Implemented vectorized operations to dramatically improve computational efficiency  

- 🎯 Feedforward Vectorization: Matrix-based implementation  
- 🔄 Backpropagation Vectorization: Eliminated iterative loops  

**📈 Enhanced Evaluation:**  
- Increased to 20 epochs for comprehensive assessment  
- Reported final model accuracy and execution time  
- Multiple executions to account for performance variability  
- Cost trajectory visualization over training  

---

### 5. 🧪 Model Testing & Evaluation  
Comprehensive performance assessment using full dataset (4 classes, 8000 samples)  

**🏋️ Training Configuration:** Optimized hyperparameters  

**📋 Evaluation Metrics:**  
- Training set accuracy 📊  
- Test set accuracy 📉  
- Comparative performance analysis  
- Learning visualization: Average cost reduction over epochs  

---

## 🎯 Key Achievements  
- ✅ Built neural network from foundational principles  
- ✅ Implemented efficient data preprocessing pipeline  
- ✅ Demonstrated dramatic performance improvement through vectorization  
- ✅ Achieved measurable accuracy on CIFAR-10 subset  
- ✅ Visualized learning process through cost reduction graphs  

---

## 📁 Repository Structure  
```bash
📦 CIFAR10-NeuralNetwork
├── 📄 README.md # Project documentation
├── 📊 data/ # Dataset handling utilities
├── 🧠 models/ # Neural network implementation
├── 📈 results/ # Performance metrics and visualizations
├── 🔬 experiments/ # Testing and evaluation scripts
└── 📜 requirements.txt # Project dependencies
```
## 🚀 How to Run  

**1. Install dependencies:**  
```bash
pip install -r requirements.txt
```
**2. Preprocess data:**
``` bash
python data/preprocessing.py
```

**3. Train model:**
``` bash
python models/train.py
```

**4. Evaluate performance:**
```bash
python experiments/evaluate.py
```
## 📊 Expected Results

* Baseline Accuracy: ~25–30% (random initialization)

* Vectorized Speedup: 5–10x performance improvement

* Final Accuracy: Measurable improvement over baseline

* Learning Curve: Consistent cost reduction across epochs

## 🔮 Future Enhancements

* Additional layer architectures

* Alternative activation functions (ReLU, tanh)

* Regularization techniques (Dropout, L2)

* Hyperparameter optimization framework

* Extension to full CIFAR-10 dataset (10 classes)
