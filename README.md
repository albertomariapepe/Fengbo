# 🚀 Fengbo: Solving 3D PDEs with Clifford Algebra

Fengbo is a deep learning pipeline built entirely in **Clifford Algebra** to solve **3D Partial Differential Equations (PDEs)**, specifically for **Computational Fluid Dynamics (CFD)**. It leverages **3D Convolutional** and **Fourier Neural Operator (FNO) layers**, making it a powerful, physics-aware, and interpretable approach to PDE modeling. 

---

## ✨ Features
✅ **Clifford Algebra-Based**: Works entirely in 3D Clifford Algebra for enhanced geometric and physics-based understanding.  
✅ **Efficient Architecture**: Uses only **42 million** trainable parameters with a streamlined design.  
✅ **Superior Accuracy**: Outperforms **5 out of 6** models reported in [Li et al. 2024](https://arxiv.org/abs/2309.00583) on the **ShapeNet Car** dataset.  
✅ **Computationally Efficient**: Reduces complexity compared to graph-based methods.  
✅ **Interpretable Outputs**: Outputs can be visualized as 3D physical quantities, making it a **white-box model**.  
✅ **Joint Estimation**: Simultaneously predicts **pressure and velocity fields**.  

---

## 📖 Method Overview
Fengbo models PDE solutions as an **interpretable mapping** from **geometry to physics**, ensuring an efficient, geometry-aware, and physics-consistent solution. It consists of:

- **3D Convolutional Layers** 🧩
- **Fourier Neural Operator (FNO) Layers** 🎛️
- **Clifford Algebra Operations** 📐

This combination allows for a direct mapping from input geometries to the corresponding physics fields, achieving both efficiency and accuracy.



## 🔧 Installation
To install and set up Fengbo, follow these steps:
```bash
# Clone the repository
git clone https://github.com/yourusername/fengbo.git
pip install -r requirements.txt
```

---

## 📜 Citation
If you use Fengbo in your research, please cite:
```
@inproceedings{pepefengbo,
  title={Fengbo: a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics},
  author={Pepe, Alberto and Montanari, Mattia and Lasenby, Joan},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
---

## 🤝 Contributing
We welcome contributions! Feel free to open issues or submit PRs.

---

## 📬 Contact
For any inquiries, reach out via email or open an issue on GitHub.

