#pragma once
#include <math.h>
#include <s21_matrix_oop.h>

#include <cassert>
#include <random>
#include <utility>

namespace nn {
inline double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }

inline double sigmoid_derivative(double x) { return (x * (1 - x)); }

inline double relu(double x) {
  if (x > 0)
    return x;
  else
    return 0;
}

class MLP {
 public:
  std::vector<size_t> units_per_layer;
  std::vector<S21Matrix> bias_vectors;
  std::vector<S21Matrix> weight_matrices;
  std::vector<S21Matrix> activations;
  float lr;

  explicit MLP(std::vector<size_t> units_per_layer, float lr = .001f)
      : units_per_layer(units_per_layer),
        weight_matrices(),
        bias_vectors(),
        activations(),
        lr(lr) {
    for (size_t i = 0; i < units_per_layer.size() - 1; ++i) {
      size_t in_channels{units_per_layer[i]};
      size_t out_channels{units_per_layer[i + 1]};
      S21Matrix b = mtx::randn(out_channels, 1);  // Bias
      S21Matrix W = mtx::randn(out_channels, in_channels);
      weight_matrices.push_back(W);
      bias_vectors.push_back(b);
      activations.resize(units_per_layer.size());
    }
  }

  S21Matrix forward(S21Matrix& x) {
    activations[0] = x;
    S21Matrix prev_out = x;
    for (int i = 0; i < weight_matrices.size(); i++) {
      prev_out = weight_matrices[i].matmul(prev_out) + bias_vectors[i];
      activations[i + 1] = prev_out.apply_function<double>(sigmoid);
    }
    return activations.back();
  }
  void backward(S21Matrix& target) {
    std::vector<S21Matrix> dW(weight_matrices.size());
    std::vector<S21Matrix> db(weight_matrices.size());
    S21Matrix error = activations.back() - target;
    S21Matrix back_act =
        activations.back().apply_function<double>(sigmoid_derivative);
    S21Matrix delta = error.multiply_elementwise(back_act);
    S21Matrix tmp_transposed = activations[activations.size() - 2].Transpose();
    dW.back() = delta.matmul(tmp_transposed);
    db.back() = delta;
    for (int i = weight_matrices.size() - 2; i >= 0; --i) {
      S21Matrix tmp_transposed = weight_matrices[i + 1].Transpose();
      S21Matrix delta_next_layer = tmp_transposed.matmul(delta);
      S21Matrix sigmoid_prime =
          activations[i + 1].apply_function<double>(sigmoid_derivative);
      delta = delta_next_layer.multiply_elementwise(sigmoid_prime);
      S21Matrix tmp_transposed2 = activations[i].Transpose();
      dW[i] = delta.matmul(tmp_transposed2);
      db[i] = delta;
    }
    for (int i = 0; i < weight_matrices.size(); ++i) {
      dW[i].MulNumber(lr);
      db[i].MulNumber(lr);
      weight_matrices[i] -= dW[i];
      bias_vectors[i] -= db[i];
    }
  }

  void print_info(bool additional = false) {
    std::cout << "MODEL info:\n";
    for (size_t i = 0; i < units_per_layer.size(); ++i) {
      std::cout << "Layer " << i << ": " << units_per_layer[i] << " neurons\n";
      if (i < weight_matrices.size()) {
        std::cout << "  Weight matrix: " << weight_matrices[i].get_cols() << "x"
                  << weight_matrices[i].get_rows() << "\n";
        if (additional) {
          std::cout << "Info of Weight matrix: \n";
          weight_matrices[i].print();
        }
      }
    }
    std::cout << "Learning rate: " << lr << "\n";
    std::cout << std::endl;
  }
  enum class LossType { MSE, CrossEntropy };

  double compute_loss(S21Matrix& prediction, S21Matrix& target,
                      LossType loss_type) {
    double loss = 0.0;
    switch (loss_type) {
      case LossType::MSE:
        loss = (prediction - target)
                   .apply_function<double>([](double x) { return x * x; })
                   .sum() /
               prediction.get_rows();
        break;
      case LossType::CrossEntropy:
        loss = -target
                    .apply_function<double>([&](double x) {
                      return x * log(prediction(0, 0) + 1e-9);
                    })
                    .sum();
        break;
      default:
        throw std::invalid_argument("Invalid loss type");
    }
    return loss;
  }
  void save_model(const std::string& file_name) {
    std::ofstream out(file_name, std::ios::binary);
    if (!out) {
      throw std::runtime_error("Unable to open file for writing");
    }

    for (auto& W : weight_matrices) {
      int rows = W.get_rows();
      int cols = W.get_cols();
      out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
      out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
      out.write(reinterpret_cast<const char*>(W.data()),
                rows * cols * sizeof(double));
    }

    for (auto& b : bias_vectors) {
      int rows = b.get_rows();
      int cols = b.get_cols();
      out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
      out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
      out.write(reinterpret_cast<const char*>(b.data()),
                rows * cols * sizeof(double));
    }

    out.close();
  }

  void load_model(const std::string& file_name) {
    std::ifstream in(file_name, std::ios::binary);
    for (auto& W : weight_matrices) {
      in.read(reinterpret_cast<char*>(W.data()),
              W.get_numel() * sizeof(double));
    }
    for (auto& b : bias_vectors) {
      in.read(reinterpret_cast<char*>(b.data()),
              b.get_numel() * sizeof(double));
    }
    in.close();
  }
};
auto make_model(size_t in_channels, size_t out_channels,
                size_t hidden_units_per_layer, int hidden_layers, float lr,
                std::string activation_functon) {
  std::vector<size_t> units_per_layer;

  units_per_layer.push_back(in_channels);

  for (int i = 0; i < hidden_layers; ++i)
    units_per_layer.push_back(hidden_units_per_layer);

  units_per_layer.push_back(out_channels);

  MLP model(units_per_layer, lr);
  return model;
}
}  // namespace nn
