AR = ar 
CC = g++
MAIN = main.cpp
INCLUDE = -I matrix/ 
OBJECTS = s21_martix_oop.o 
BIN = main
RM = rm -rf 

all: run

#nn.o: s21_martix_oop.o
#	$(CC) -c  nn.h -o $@ $(INCLUDE) $< 

s21_martix_oop.o:
	cd matrix/ && make $@ && mv $@ ../

#%.o: %.c
#    $(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJECTS)
	$(CC) $(MAIN) -o main $(INCLUDE) $<

run: clean $(BIN) 
	./main

clean:
	$(RM) $(OBJECTS)
	$(RM) $(BIN)