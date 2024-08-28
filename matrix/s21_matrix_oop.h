#pragma once
#include <functional>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>

class S21Matrix {
 public:
  S21Matrix();
  S21Matrix(int row_col);
  S21Matrix(int rows, int cols);
  S21Matrix(int rows, int cols, std::initializer_list<double> data);
  S21Matrix(const S21Matrix& other);
  S21Matrix& operator=(const S21Matrix& other);
  S21Matrix(S21Matrix&& other);
  S21Matrix& operator=(S21Matrix&& other);
  void set_cols(int number);
  void set_rows(int number);
  void set_dims(int rows, int cols);
  ~S21Matrix();

  void print() { std::cout << (*this) << "\n"; }
  void fill_matrix(double numbers);
  void byte_matrix(int row, int col, S21Matrix& result);

  double& operator()(int rows, int cols);
  double& operator()(int rows, int cols) const;
  friend std::ostream& operator<<(std::ostream& os, const S21Matrix& matrix) {
    for (int i = 0; i < matrix.rows_; i++) {
      for (int j = 0; j < matrix.cols_; j++) {
        os << matrix.matrix_[i * matrix.cols_ + j] << " ";
      }
      os << "\n";
    }
    return os;
  }

  bool EqMatrixTol(const S21Matrix& other);
  S21Matrix operator+(const S21Matrix& o) const;
  S21Matrix operator*(const S21Matrix& o) const;
  S21Matrix operator-(const S21Matrix& o) const;
  S21Matrix& operator*=(const S21Matrix& o);
  S21Matrix& operator+=(const S21Matrix& o);
  S21Matrix& operator-=(const S21Matrix& o);
  bool operator==(const S21Matrix& other);
  void MulNumber(const double num);
  bool EqMatrix(const S21Matrix& other);
  void SumMatrix(const S21Matrix& other);
  void MulMatrix(const S21Matrix& other);
  void SubMatrix(const S21Matrix& other);
  int get_rows();   // returning rows
  int get_cols();   // returning cols
  int get_numel();  // returning cols*rows

  double mean(S21Matrix& mat);
  double sum();

  double Determinant();
  S21Matrix Transpose();
  S21Matrix CalcComplements();
  S21Matrix InverseMatrix();
  void fill_(double num);

  S21Matrix multiply_elementwise(S21Matrix& target);
  S21Matrix matmul(S21Matrix& target);
  S21Matrix addM(S21Matrix& target);
  S21Matrix slice_row(int row_index) const;

  double* data();
  const double* data() const;

  template <typename T>
  S21Matrix apply_function(const std::function<T(const T&)>& function) {
    S21Matrix output = S21Matrix(get_rows(), get_cols());
    for (int r = 0; r < get_rows(); ++r) {
      for (int c = 0; c < get_cols(); ++c) {
        output(r, c) = function((*this)(r, c));
      }
    }
    return output;
  }

 private:
  int rows_, cols_;
  double* matrix_;
  int numel = rows_ * cols_;
};

struct mtx {
  static S21Matrix zeros(int rows, int cols);
  static S21Matrix ones(int rows, int cols);
  static S21Matrix randn(int rows, int cols);
  static S21Matrix rand(size_t rows, size_t cols);
};
