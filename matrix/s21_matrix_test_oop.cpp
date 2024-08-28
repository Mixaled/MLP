#include <gtest/gtest.h>

#include "s21_matrix_oop.h"

TEST(Matrix_test, CreateMatrix) {
  S21Matrix a = S21Matrix(3, 3);
  S21Matrix b = S21Matrix(3);
  EXPECT_EQ(a.get_rows(), 3);
  EXPECT_EQ(b.get_cols(), 3);
}

TEST(CreateMatrixTest, Valid) {
  S21Matrix a(3, 3);

  EXPECT_EQ(a.get_rows(), 3);
}

TEST(CreateMatrixTest, InvalidRows) {
  EXPECT_THROW({ S21Matrix a(-1, 3); }, std::runtime_error);
}

TEST(CreateMatrixTest, InvalidColumns) {
  EXPECT_THROW({ S21Matrix a(3, -1); }, std::runtime_error);
}

TEST(ComplementsTest, NonSquare) {
  S21Matrix matrix(2, 3);
  EXPECT_THROW(matrix.CalcComplements(), std::runtime_error);
}

TEST(ComplementsTest, Matrix1x1) {
  S21Matrix matrix(1, 1, {5});
  S21Matrix expected(1, 1, {1});
  std::cout << matrix;
  EXPECT_TRUE(matrix.CalcComplements().EqMatrixTol(expected));
}

TEST(ComplementsTest, Matrix2x2) {
  S21Matrix matrix(2, 2, {1, 2, 3, 4});
  S21Matrix expected(2, 2, {4, -3, -2, 1});
  EXPECT_TRUE(matrix.CalcComplements().EqMatrixTol(expected));
}

TEST(ComplementsTest, Matrix3x3) {
  S21Matrix matrix(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  S21Matrix expected(3, 3, {-3, 6, -3, 6, -12, 6, -3, 6, -3});
  EXPECT_TRUE(matrix.CalcComplements().EqMatrixTol(expected));
}

TEST(ComplementsTest, Matrix4x4) {
  S21Matrix matrix(4, 4,
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  S21Matrix expected(4, 4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  EXPECT_TRUE(matrix.CalcComplements().EqMatrixTol(expected));
}

TEST(DeterminantTest, InvalidMatrix) {
  EXPECT_THROW(
      {
        S21Matrix null_matrix;
        double det = null_matrix.Determinant();
      },
      std::runtime_error);
}

TEST(DeterminantTest, NonSquareMatrix) {
  S21Matrix mat(3, 4,
                {
                    1, 2, 3, 4,  /**/
                    4, 5, 6, 7,  /**/
                    7, 8, 9, 10, /**/
                });

  EXPECT_THROW({ double det = mat.Determinant(); }, std::runtime_error);
}

TEST(DeterminantTest, Matrix1x1) {
  S21Matrix matrix(1, 1, {5.0});
  EXPECT_DOUBLE_EQ(matrix.Determinant(), 5.0);
}

TEST(DeterminantTest, Matrix2x2) {
  S21Matrix matrix(2, 2, {4, 7, 2, 6});
  EXPECT_DOUBLE_EQ(matrix.Determinant(), 10.0);
}

TEST(DeterminantTest, Matrix3x3) {
  S21Matrix matrix(3, 3,
                   {
                       1, 2, 3, /**/
                       0, 1, 4, /**/
                       5, 6, 0, /**/
                   });
  EXPECT_DOUBLE_EQ(matrix.Determinant(), 1.0);
}

TEST(EqMatrixTest, Equal) {
  S21Matrix a(3, 3,
              {
                  1, 2, 3, /**/
                  4, 5, 6, /**/
                  7, 8, 9, /**/
              });

  S21Matrix b(3, 3,
              {
                  1, 2, 3, /**/
                  4, 5, 6, /**/
                  7, 8, 9, /**/
              });

  EXPECT_TRUE(a == b);
}

TEST(EqMatrixTest, Different) {
  S21Matrix a(3, 3,
              {
                  1, 2, 3, /**/
                  4, 5, 6, /**/
                  7, 8, 9, /**/
              });

  S21Matrix b(3, 3,
              {
                  1, 2, 3, /**/
                  4, 5, 6, /**/
                  7, 8, 0, /**/
              });

  EXPECT_FALSE(a == b);
}

TEST(EqMatrixTest, DifferentSizes) {
  S21Matrix a(3, 3,
              {
                  1, 2, 3, /**/
                  4, 5, 6, /**/
                  7, 8, 9, /**/
              });

  S21Matrix b(3, 4,
              {
                  1, 2, 3, 4,  /**/
                  4, 5, 6, 7,  /**/
                  7, 8, 9, 10, /**/
              });

  EXPECT_FALSE(a == b);
}

TEST(EqMatrixTest, ZeroSizeEq) {
  S21Matrix a(0, 0), b(0, 0);
  EXPECT_TRUE(a == b);
}

TEST(EqMatrixTest, ZeroSizeNeqRows) {
  S21Matrix a(0, 0), b(500, 0);
  EXPECT_FALSE(a == b);
}

TEST(EqMatrixTest, ZeroSizeNeqCols) {
  S21Matrix a(0, 500), b(0, 0);
  EXPECT_FALSE(a == b);
}

TEST(InverseMatrixTest, Invalid) {
  S21Matrix null_matrix;
  EXPECT_THROW({ auto det = null_matrix.InverseMatrix(); }, std::runtime_error);
}

TEST(InverseMatrixTest, InvalidSize) {
  S21Matrix a(3, 2);

  EXPECT_THROW({ auto det = a.InverseMatrix(); }, std::runtime_error);
}

TEST(InverseMatrixTest, Singular) {
  S21Matrix source(2, 2,
                   {
                       1, 2, /**/
                       2, 4, /**/
                   });
  EXPECT_THROW(source.InverseMatrix(), std::runtime_error);
}

TEST(InverseMatrixTest, Valid1) {
  S21Matrix source(2, 2,
                   {
                       4, 7, /**/
                       2, 6, /**/
                   });

  S21Matrix expected(2, 2,
                     {
                         0.6, -0.7, /**/
                         -0.2, 0.4, /**/
                     });

  EXPECT_TRUE(source.InverseMatrix().EqMatrixTol(expected));
}

TEST(InverseMatrixTest, Valid2) {
  double expected_result[][3] = {};

  S21Matrix source(3, 3,
                   {
                       1, 2, 3, /**/
                       0, 1, 4, /**/
                       5, 6, 0, /**/
                   });

  S21Matrix expected(3, 3,
                     {
                         -24, 18, 5,  /**/
                         20, -15, -4, /**/
                         -5, 4, 1,    /**/
                     });

  EXPECT_TRUE(source.InverseMatrix().EqMatrixTol(expected));
}

TEST(InverseMatrixTest, Valid3) {
  S21Matrix source(3, 3,
                   {
                       2, 0, 0, /**/
                       0, 3, 0, /**/
                       0, 0, 4, /**/
                   });

  S21Matrix expected(3, 3,
                     {
                         1.0 / 2.0, 0, 0, /**/
                         0, 1.0 / 3.0, 0, /**/
                         0, 0, 1.0 / 4.0, /**/
                     });

  EXPECT_TRUE(source.InverseMatrix().EqMatrixTol(expected));
}

TEST(InverseMatrixTest, Valid4) {
  S21Matrix source(2, 2,
                   {
                       3, 0, /**/
                       0, 3, /**/
                   });

  S21Matrix expected(2, 2,
                     {
                         1.0 / 3.0, 0, /**/
                         0, 1.0 / 3.0, /**/
                     });

  EXPECT_TRUE(source.InverseMatrix().EqMatrixTol(expected));
}

TEST(MultMatrixTest, Valid) {
  S21Matrix a(1, 3, {1, 2, 3});
  S21Matrix b(3, 1, {4, 5, 6});

  S21Matrix expected(1, 1, {32});

  EXPECT_TRUE((a * b).EqMatrixTol(expected));
}

TEST(MultMatrixTest, Invalid) {
  S21Matrix a(1, 3, {1, 2, 3});
  S21Matrix b(2, 2, {1, 2, 3, 4});

  EXPECT_THROW(a * b, std::runtime_error);
}

TEST(MultMatrixTest, Valid2) {
  S21Matrix a(8, 1, {10, 7, 6, 9, 6, 6, 5, 2});
  S21Matrix b(1, 10, {6, 8, 5, 4, 4, 5, 9, 7, 1, 10});

  S21Matrix expected(8, 10, {60, 80, 50, 40, 40, 50, 90, 70, 10, 100, /**/
                             42, 56, 35, 28, 28, 35, 63, 49, 7,  70,  /**/
                             36, 48, 30, 24, 24, 30, 54, 42, 6,  60,  /**/
                             54, 72, 45, 36, 36, 45, 81, 63, 9,  90,  /**/
                             36, 48, 30, 24, 24, 30, 54, 42, 6,  60,  /**/
                             36, 48, 30, 24, 24, 30, 54, 42, 6,  60,  /**/
                             30, 40, 25, 20, 20, 25, 45, 35, 5,  50,  /**/
                             12, 16, 10, 8,  8,  10, 18, 14, 2,  20,
                             /**/});

  EXPECT_TRUE((a * b).EqMatrixTol(expected));
}

TEST(MultMatrixTest, Valid3) {
  S21Matrix a(3, 5, {5, 6, 1, 8, 10, 1, 10, 7, 2, 4, 5, 4, 8, 4, 1});
  S21Matrix b(5, 2, {4, 5, 4, 4, 2, 8, 6, 4, 9, 6});

  S21Matrix expected(3, 2, {184, 149, 106, 133, 85, 127});

  EXPECT_TRUE((a * b).EqMatrixTol(expected));
}

TEST(SizesTest, Valid1) {
  S21Matrix a(2, 2,
              {
                  1, 2, /**/
                  3, 4, /**/
              });

  S21Matrix expected(2, 4,
                     {
                         1, 2, 0, 0, /**/
                         3, 4, 0, 0, /**/
                     });

  a.set_cols(4);

  EXPECT_TRUE(a.EqMatrixTol(expected));
}

TEST(SizesTest, Valid2) {
  S21Matrix a(2, 2,
              {
                  1, 2, /**/
                  3, 4, /**/
              });

  S21Matrix expected(4, 2,
                     {
                         1, 2, /**/
                         3, 4, /**/
                         0, 0, /**/
                         0, 0, /**/
                     });

  a.set_rows(4);

  EXPECT_TRUE(a.EqMatrixTol(expected));
}

TEST(SizesTest, Valid3) {
  S21Matrix a(2, 2,
              {
                  1, 2, /**/
                  3, 4, /**/
              });

  S21Matrix expected(2, 1,
                     {
                         1, /**/
                         3, /**/
                     });

  a.set_cols(1);

  EXPECT_TRUE(a.EqMatrixTol(expected));
}

TEST(SizesTest, Valid4) {
  S21Matrix a(2, 2,
              {
                  1, 2, /**/
                  3, 4, /**/
              });

  S21Matrix expected(1, 2,
                     {
                         1, 2, /**/
                     });

  a.set_rows(1);

  EXPECT_TRUE(a.EqMatrixTol(expected));
}

TEST(SumTest, InvalidSizes) {
  S21Matrix a(2, 3);
  S21Matrix b(3, 2);
  EXPECT_THROW(a + b, std::runtime_error);
}

TEST(SumTest, Valid) {
  S21Matrix a(2, 2, {1, 2, 3, 4});
  S21Matrix b(2, 2, {5, 6, 7, 8});

  S21Matrix expected(2, 2, {6, 8, 10, 12});
  EXPECT_TRUE((a + b).EqMatrixTol(expected));
}

TEST(SubTest, InvalidSizes) {
  S21Matrix a(2, 3);
  S21Matrix b(3, 2);
  EXPECT_THROW(a - b, std::runtime_error);
}

TEST(SubTest, Valid) {
  S21Matrix a(2, 2, {1, 2, 3, 4});
  S21Matrix b(2, 2, {5, 6, 7, 8});

  S21Matrix expected(2, 2, {-4, -4, -4, -4});
  EXPECT_TRUE((a - b).EqMatrixTol(expected));
}

TEST(SumTest, BigMatrices) {
  S21Matrix a(
      7, 7,
      {
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
          35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
      });
  S21Matrix b(
      7, 7,
      {
          49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
          32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
          15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,
      });

  S21Matrix expected(
      7, 7,
      {
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      });

  EXPECT_TRUE((a + b).EqMatrixTol(expected));
}

TEST(TransposeTest, Valid1) {
  S21Matrix source(2, 2, {1, 2, 3, 4});
  S21Matrix expected(2, 2, {1, 3, 2, 4});

  EXPECT_TRUE(source.Transpose().EqMatrixTol(expected));
}

TEST(TransposeTest, Valid2) {
  S21Matrix source(2, 3, {1, 2, 3, 4, 5, 6});
  S21Matrix expected(3, 2, {1, 4, 2, 5, 3, 6});

  EXPECT_TRUE(source.Transpose().EqMatrixTol(expected));
}

TEST(TransposeTest, Valid3) {
  S21Matrix source(2, 3, {1, 2, 3, 4, 5, 6});
  S21Matrix expected(3, 2, {1, 4, 2, 5, 3, 6});

  EXPECT_TRUE(source.Transpose().EqMatrixTol(expected));
}

TEST(TransposeTest, Valid4) {
  S21Matrix source(3, 2, {1, 4, 2, 5, 3, 6});
  S21Matrix expected(2, 3, {1, 2, 3, 4, 5, 6});

  EXPECT_TRUE(source.Transpose().EqMatrixTol(expected));
}

TEST(TransposeTest, Valid5) {
  S21Matrix source(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1});

  EXPECT_TRUE(source.Transpose().EqMatrixTol(source));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}