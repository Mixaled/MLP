#include "s21_matrix_oop.h"

int main() {
  S21Matrix mama = S21Matrix(2, 2);
  mama(1, 1) = 10.0;
  mama(0, 0) = 10.0;
  S21Matrix papa = S21Matrix(2);
  papa(1, 1) = 4.0;
  mama(0, 0) = 10.0;
  papa = papa * mama;
  papa.print();
  papa.MulMatrix(mama);
  papa.print();
  S21Matrix papa_clone = papa;
  bool res = papa.EqMatrix(papa_clone);
  std::cout << res << "\n";
  std::cout << (papa == papa_clone) << "\n";
  S21Matrix xM = S21Matrix(2, 2);
  xM.fill_matrix(9.0);
  xM.print();
  S21Matrix yM = S21Matrix(2, 2);
  yM.fill_matrix(9.0);
  yM.MulMatrix(xM);
  yM.print();
  S21Matrix matrix = S21Matrix(3, 3);
  matrix(0, 0) = 1.0;
  matrix(0, 1) = 2.0;
  matrix(0, 2) = 2.0;
  matrix(1, 0) = 3.0;
  matrix(1, 1) = 2.0;
  matrix(1, 2) = 1.0;
  matrix(2, 0) = 1.0;
  matrix(2, 1) = 1.0;
  matrix(2, 2) = 3.0;
  double determinant = matrix.Determinant();
  std::cout << "The determinant of this matrix is: " << determinant
            << std::endl;
  matrix.print();
  S21Matrix matrix2 = S21Matrix(2, 2);
  matrix2(0, 0) = 1.0;
  matrix2(0, 1) = 2.0;
  matrix2(1, 0) = 4.0;
  matrix2(1, 1) = 5.0;
  std::cout << "Original matrix:"
            << "\n";
  matrix2.print();
  S21Matrix transposed = matrix2.Transpose();
  std::cout << "Transposed matrix:"
            << "\n";
  transposed.print();
  S21Matrix matrix3 = S21Matrix(2, 2);
  matrix3(0, 0) = 1.0;
  matrix3(0, 1) = 2.0;
  matrix3(1, 0) = 3.0;
  matrix3(1, 1) = 4.0;
  std::cout << "Original matrix:"
            << "\n";
  matrix3.print();
  S21Matrix calc_comp = matrix3.CalcComplements();
  std::cout << "Calc_complements:"
            << "\n";
  calc_comp.print();
  S21Matrix matrix4 = S21Matrix(2, 2);
  matrix4(0, 0) = 1.0;
  matrix4(0, 1) = 2.0;
  matrix4(1, 0) = 3.0;
  matrix4(1, 1) = 4.0;
  std::cout << "Original matrix:"
            << "\n";
  matrix4.print();
  S21Matrix inverse = matrix4.InverseMatrix();
  std::cout << "Inverse matrix:"
            << "\n";
  inverse.print();
  S21Matrix check_index(1);
  double a = check_index(0, 0);
  std::cout << a << "!\n";
  check_index.print();
  return 0;
}