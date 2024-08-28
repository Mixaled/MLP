#include "s21_matrix_oop.h"

S21Matrix::S21Matrix() : rows_(0), cols_(0), matrix_(nullptr) {
  numel = 0;
};  // default constructor

int S21Matrix::get_numel() { return numel; }

S21Matrix::S21Matrix(int row_col) : S21Matrix(row_col, row_col) {};
S21Matrix::S21Matrix(int rows, int cols)
    : rows_(rows > 0 ? rows : 0), cols_(cols > 0 ? cols : 0), matrix_(nullptr) {
  numel = rows_ * cols_;
  if (rows < 0 || cols < 0) {
    throw std::runtime_error("ERR: negative index");
  }
  if (rows_ > 0 && cols_ > 0) {
    matrix_ = new double[cols_ * rows_];
    for (int i = 0; i < cols_ * rows_; i++) {
      matrix_[i] = 0;
    }
  }
}

double* S21Matrix::data() { return matrix_; }
const double* S21Matrix::data() const { return matrix_; }

S21Matrix::S21Matrix(const S21Matrix& other)
    : S21Matrix(other.rows_, other.cols_) {  // copy constructor
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      matrix_[i * cols_ + j] = other.matrix_[i * cols_ + j];
    }
  }
}

S21Matrix::S21Matrix(int rows, int cols, std::initializer_list<double> data)
    : S21Matrix(rows, cols) {
  if (data.size() != (long unsigned int)(rows * cols)) {
    throw std::runtime_error("ERR stupid data");
  }
  std::copy(data.begin(), data.end(), matrix_);
}

bool S21Matrix::EqMatrixTol(const S21Matrix& o) {
  if (rows_ != o.rows_ || cols_ != o.cols_) return false;
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      if ((std::abs((*this)(i, j) - o(i, j))) > 1.e05) return false;
    }
  }
  return true;
}

int S21Matrix::get_rows() { return rows_; }

int S21Matrix::get_cols() { return cols_; }

// Оператор присваивания копированием
S21Matrix& S21Matrix::operator=(const S21Matrix& other) {
  if (this != &other) {
    delete[] matrix_;
    rows_ = other.rows_;
    cols_ = other.cols_;
    matrix_ = new double[rows_ * cols_];
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        matrix_[i * cols_ + j] = other.matrix_[i * cols_ + j];
      }
    }
  }
  return *this;
}

bool S21Matrix::operator==(const S21Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) return false;
  for (int i = 0; i < rows_; i++) {
    for (int c = 0; c < cols_; c++) {
      if ((*this)(i, c) != other(i, c)) return false;
    }
  }
  return true;
}

// Конструктор перемещения
S21Matrix::S21Matrix(S21Matrix&& other)
    : rows_(other.rows_), cols_(other.cols_), matrix_(other.matrix_) {
  other.rows_ = 0;
  other.cols_ = 0;
  other.matrix_ = nullptr;
}
// Оператор присваиваниия перемещением
S21Matrix& S21Matrix::operator=(S21Matrix&& other) {
  if (this != &other) {
    delete[] matrix_;
    rows_ = other.rows_;
    cols_ = other.cols_;
    matrix_ = other.matrix_;
    other.rows_ = 0;
    other.cols_ = 0;
    other.matrix_ = nullptr;
  }
  return *this;
}

double& S21Matrix::operator()(int rows, int cols) {
  // std::cout << "Rows_matrix: " << rows_ << " Rows_in: " << rows << "\n";
  if (rows >= rows_ || cols >= cols_) {
    throw std::runtime_error("ERR: indx error operator (1)");
  }
  return matrix_[rows * cols_ + cols];
}

double& S21Matrix::operator()(int rows, int cols) const {
  if (rows >= rows_ || cols >= cols_) {
    throw std::runtime_error("ERR: indx error operator (2)");
  }
  return matrix_[rows * cols_ + cols];
}
S21Matrix S21Matrix::operator+(const S21Matrix& o) const {
  S21Matrix result(*this);
  result += o;
  return result;
}

S21Matrix S21Matrix::operator*(const S21Matrix& o) const {
  S21Matrix result(*this);
  result *= o;
  return result;
}

S21Matrix S21Matrix::operator-(const S21Matrix& o) const {
  S21Matrix result(*this);
  result -= o;
  return result;
}

S21Matrix& S21Matrix::operator+=(const S21Matrix& o) {
  if (o.rows_ != rows_ || o.cols_ != cols_) {
    throw std::runtime_error("ERR: rows and cols not equal");
  }
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      (*this)(i, j) += o(i, j);
    }
  }
  return *this;
}

S21Matrix& S21Matrix::operator*=(const S21Matrix& o) {
  if (cols_ != o.rows_) {
    throw std::runtime_error("ERR: rows and cols not equal");
  }
  S21Matrix result(rows_, o.cols_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < o.cols_; j++) {
      double sum = 0;
      for (int k = 0; k < cols_; k++) {
        sum += (*this)(i, k) * o(k, j);
      }
      result(i, j) = sum;
    }
  }
  *this = result;
  return *this;
}

S21Matrix& S21Matrix::operator-=(const S21Matrix& o) {
  if (o.rows_ != rows_ || o.cols_ != cols_) {
    throw std::runtime_error("ERR: rows and cols not equal (-=)");
  }
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      (*this)(i, j) -= o(i, j);
    }
  }
  return *this;
}

void S21Matrix::fill_matrix(double n) {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      (*this)(i, j) = n;
    }
  }
}

// создания подматрицы путем удаления указанной строки и столбца из исходной
// матрицы
void S21Matrix::byte_matrix(int row, int col, S21Matrix& result) {
  int rows = rows_ - 1;
  int cols = cols_ - 1;
  result = S21Matrix(rows, cols);
  for (int i = 0, m = 0; i < rows_; i++) {
    if (i == row) {
      continue;
    } else {
      for (int j = 0, n = 0; j < cols_; j++) {
        if (j == col) {
          continue;
        } else {
          result(m, n) = (*this)(i, j);
          n++;
        }
      }
      m++;
    }
  }
}

double S21Matrix::Determinant() {
  double result = 0.0;
  if (rows_ <= 0 || cols_ <= 0) {
    throw std::runtime_error("Matrix incorrect");
  }
  if (rows_ != cols_) {
    throw std::runtime_error("ERR: matrix is not square");
  }
  if (rows_ == 1) {
    result = (*this)(0, 0);
  } else if (cols_ == 2) {
    result = (*this)(0, 0) * (*this)(1, 1) - (*this)(1, 0) * (*this)(0, 1);
  } else {
    int row = rows_;
    S21Matrix temp;
    int minus = 1;
    double det = 0;
    for (int i = 0; i < row; i++) {
      byte_matrix(0, i, temp);
      det = temp.Determinant();
      // std::cout << result << "\n";
      result += (*this)(0, i) * det * minus;
      minus = -minus;
    }
  }
  return result;
}

S21Matrix S21Matrix::Transpose() {
  if (rows_ <= 0 || cols_ <= 0) {
    throw std::runtime_error("Matrix incorrect");
  }
  S21Matrix result(cols_, rows_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

void S21Matrix::SumMatrix(const S21Matrix& o) {
  if (o.rows_ != rows_ || o.cols_ != cols_) {
    throw std::runtime_error("ERR: rows or cols not equal");
  }
  (*this) = (*this) + o;
}

void S21Matrix::SubMatrix(const S21Matrix& o) {
  if (o.rows_ != rows_ || o.cols_ != cols_) {
    throw std::runtime_error("ERR: rows or cols not equal");
  }
  (*this) = (*this) - o;
}

void S21Matrix::set_rows(int rows) { set_dims(rows, get_cols()); }

void S21Matrix::set_cols(int cols) { set_dims(get_rows(), cols); }

void S21Matrix::set_dims(int rows, int cols) {
  S21Matrix matrix(rows, cols);
  for (int i = 0; i < std::min(rows_, rows); i++) {
    for (int j = 0; j < std::min(cols_, cols); j++) {
      matrix(i, j) = (*this)(i, j);
    }
  }
  (*this) = matrix;
}

void S21Matrix::MulMatrix(const S21Matrix& o) { (*this) = (*this) * o; }

void S21Matrix::MulNumber(const double num) {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      (*this)(i, j) *= num;
    }
  }
}
bool S21Matrix::EqMatrix(const S21Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) return false;
  for (int i = 0; i < rows_; i++) {
    for (int c = 0; c < cols_; c++) {
      if ((*this)(i, c) != other(i, c)) return false;
    }
  }
  return true;
}

S21Matrix S21Matrix::CalcComplements() {
  if (rows_ <= 0 || cols_ <= 0) {
    throw std::runtime_error("ERR: incorrect matrix");
  }
  if (rows_ != cols_) {
    throw std::runtime_error("ERR: matrix is not square");
  }
  S21Matrix result(rows_, cols_);
  if (rows_ == 1) {
    result(0, 0) = 1;
    return result;
  }
  S21Matrix temp(rows_, cols_);
  double minor, minus;
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      byte_matrix(i, j, temp);
      minor = temp.Determinant();
      minus = ((i + j) % 2) ? -1 : 1;
      result(i, j) = minor * minus;
    }
  }
  return result;
}

S21Matrix S21Matrix::multiply_elementwise(S21Matrix& target) {
  S21Matrix output((*this));
  for (int r = 0; r < rows_; ++r) {
    for (int c = 0; c < cols_; ++c) {
      output(r, c) = target(r, c) * (*this)(r, c);
    }
  }
  return output;
}

S21Matrix S21Matrix::matmul(S21Matrix& target) {
  // assert(cols == target.rows);
  // if (cols_ == target.get_rows()) {
  //  throw("Alo mama alo");
  //}
  S21Matrix output(rows_, target.get_cols());

  for (int r = 0; r < output.get_rows(); ++r) {
    for (int c = 0; c < output.get_cols(); ++c) {
      for (int k = 0; k < target.get_rows(); ++k)
        output(r, c) += (*this)(r, k) * target(k, c);
    }
  }
  return output;
}

S21Matrix S21Matrix::addM(S21Matrix& target) {
  S21Matrix output = S21Matrix(rows_, target.cols_);

  for (int r = 0; r < output.get_rows(); ++r) {
    for (int c = 0; c < output.get_cols(); ++c) {
      output(r, c) = (*this)(r, c) + target(r, c);
    }
  }
  return output;
}
double S21Matrix::mean(S21Matrix& mat) {
  double sum = 0.0;
  int total_elements = mat.get_cols() * mat.get_rows();
  for (int i = 0; i < mat.get_rows(); ++i) {
    for (int j = 0; j < mat.get_cols(); ++j) {
      sum += mat(i, j);
    }
  }
  return sum / total_elements;
}

double S21Matrix::sum() {
  double sum = 0.0;
  for (int i = 0; i < this->get_rows(); ++i) {
    for (int j = 0; j < this->get_cols(); ++j) {
      sum += (*this)(i, j);
    }
  }
  return sum;
}

S21Matrix S21Matrix::InverseMatrix() {
  if (rows_ <= 0 || cols_ <= 0) {
    throw std::runtime_error("ERR: incorrect matrix");
  }
  double det = (*this).Determinant();
  if (det == 0.0) {
    throw std::runtime_error("CALC_ERR: determinant is 0");
  }
  S21Matrix result = (*this).CalcComplements();
  result = result.Transpose();
  result.MulNumber((1 / det));
  return result;
}
S21Matrix::~S21Matrix() { delete[] matrix_; };  // destructor

void S21Matrix::fill_(double num) {
  for (int i = 0; i < rows_; i++) {
    for (int c = 0; c < cols_; c++) {
      (*this)(i, c) = num;
    }
  }
}

S21Matrix S21Matrix::slice_row(int row_index) const {
  if (row_index < 0 || row_index >= rows_) {
    throw std::out_of_range("Row index out of range");
  }
  S21Matrix sliced(1, cols_);
  for (int c = 0; c < cols_; ++c) {
    sliced(0, c) = (*this)(row_index, c);
  }
  return sliced;
}

S21Matrix mtx::zeros(int rows, int cols) {
  S21Matrix M = S21Matrix(rows, cols);
  M.fill_(0.0);
  return M;
}

S21Matrix mtx::ones(int rows, int cols) {
  S21Matrix M = S21Matrix(rows, cols);
  M.fill_(1.0);
  return M;
}
S21Matrix mtx::randn(int rows, int cols) {
  S21Matrix M = S21Matrix(rows, cols);

  std::random_device rd{};
  std::mt19937 gen{rd()};
  double n(M.get_numel());
  double stdev{1 / sqrt(n)};
  std::normal_distribution<double> d{0, stdev};
  for (int r = 0; r < M.get_rows(); ++r) {
    for (int c = 0; c < M.get_cols(); ++c) {
      M(r, c) = d(gen);
    }
  }
  return M;
}
S21Matrix mtx::rand(size_t rows, size_t cols) {
  S21Matrix M = S21Matrix(rows, cols);

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<double> d{0, 1};

  for (int r = 0; r < M.get_rows(); ++r) {
    for (int c = 0; c < M.get_cols(); ++c) {
      M(r, c) = d(gen);
    }
  }
  return M;
}
