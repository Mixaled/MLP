#include <s21_matrix_oop.h>

#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn.h"

double step_function(double x) { return x > 0 ? 1.0 : 0.0; }

S21Matrix data(int index, int expected) {
  S21Matrix input = S21Matrix(2, 1);
  input(0, 0) = 1;
  input(1, 0) = 0;

  S21Matrix expect = S21Matrix(1, 1);
  expect(0, 0) = 0.8;

  S21Matrix input2 = S21Matrix(2, 1);
  input2(0, 0) = 0;
  input2(1, 0) = 1;

  S21Matrix expect2 = S21Matrix(1, 1);
  expect2(0, 0) = 0.8;

  S21Matrix input3 = S21Matrix(2, 1);
  input3(0, 0) = 0;
  input3(1, 0) = 0;

  S21Matrix expect3 = S21Matrix(1, 1);
  expect3(0, 0) = 0.1;

  S21Matrix input4 = S21Matrix(2, 1);
  input4(0, 0) = 1;
  input4(1, 0) = 1;

  S21Matrix expect4 = S21Matrix(1, 1);
  expect4(0, 0) = 0.8;

  std::vector<S21Matrix> res = {input, input2, input3, input4};
  std::vector<S21Matrix> exp = {expect, expect2, expect3, expect4};
  S21Matrix ret = (expected == 1) ? exp[index] : res[index];
  return ret;
}

int main() {
  nn::MLP model({2, 1}, 0.5f);
  model.print_info(true);
  int common_digit = 0;
  int max_digit = 3;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(0, 3);
  for (int i = 0; i <= 200; i++) {
    std::cout << "\nEpoch " << i << std::endl;
    S21Matrix prikol = data(common_digit, 0);
    std::cout << "INPUT: " << prikol << std::endl;
    S21Matrix predict = model.forward(prikol);
    std::cout << "PREDICTION: " << predict << std::endl;
    S21Matrix target = data(common_digit, 1);
    std::cout << "EXPECTED: " << target << std::endl;

    std::cout << "Loss: "
              << model.compute_loss(predict, target,
                                    nn::MLP::LossType::CrossEntropy)
              << std::endl;
    model.backward(target);
    common_digit = distr(gen);
  }

  int digi = 0;
  for (int i = 0; i <= 3; i++) {
    S21Matrix tet = data(digi, 0);
    double m = model.forward(tet)(0, 0);
    std::cout << "Input: " << tet(0, 0) << ' ' << tet(1, 0) << " Predict: " << m
              << "  Rounded: " << round(m)
              << " Correct: " << round(data(digi, 1)(0, 0)) << '\n';

    digi++;
  }
}