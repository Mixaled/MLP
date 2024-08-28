CC = g++
W = -Wall -Wextra -Werror -std=c++17
LIB_A = s21_matrix_oop.a
RM = rm -rf 
LIB_TEST = -lgtest -lgmock -pthread


all: test

s21_martix_oop.o: s21_matrix_oop.cpp
	$(CC) ${W} -c  s21_matrix_oop.cpp -o s21_martix_oop.o

$(LIB_A): s21_martix_oop.o
	ar -rc $(LIB_A) s21_martix_oop.o

play: clean $(LIB_A) 
	$(CC) ${W} s21_playground.cpp -o play $(LIB_A) 
	./play

test: clean $(LIB_A) 
	$(CC) s21_matrix_test_oop.cpp $(LIB_A) $(LIB_TEST) -o testbin
	./testbin

clean:
	${RM} $(LIB_A)
	${RM} s21_martix_oop.o
	${RM} main
	${RM} main.exe
	${RM} play
	${RM} play.exe
	${RM} testbin