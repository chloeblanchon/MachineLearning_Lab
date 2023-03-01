#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:25:04 2023

@author: chloe
"""

# Exercise 1.4.1
print("Exercise 1.4.1")

for i in range(1, 5):
    for j in range(i):
        print("X", end="") # end put a space between the letter x that we print i times
    print()
for i in range(4, 0, -1):
    for j in range(i):
        print("x", end="")  # we do the same thing but i decreases
    print()
        
print()  

# The first loop for loops through the numbers from 1 to 5.
# On each iteration of this loop, a second for loop is executed, which loops through the numbers from 0 to i (where i is the current loop variable).
# On each iteration of this loop, the character "x" is printed without a newline using the print function with the parameter set to "" .
# After this second loop, an empty line is printed to create a new line.
# This process is repeated for values ​​from 1 to 5.
# The second part of the code is similar, but the first loop loops through the numbers from 4 to 1 using a -1 step.
# This way, the lines are printed in reverse order to give the same end result.


# Exercise 1.4.2
print("Exercise 1.4.2")

input_str = "n45as29@#8ss6"

# This code defines a sum_num function that takes as input a string input_str.
# The function returns the sum of the digits contained in this string.

# The if statement checks whether each character is a digit using the isdigit() method.
# If so, the character is converted to an integer using the int() function and added to the sum_digit variable.


def sum_num(input_str):
    sum_digit = 0
    for x in input_str: # we go across the input_str
        if x.isdigit() == True: # we detect if the caracter is a string or a number and we convert it in a digit
            z = int(x)
            sum_digit = sum_digit + z # we sum the number

    return sum_digit

print(sum_num(input_str))

print() 

# Exercise 1.4.3
print("Exercise 1.4.3")

def decimalToBinary(val):
    if val >= 1:
        decimalToBinary(val // 2)
    print(val % 2, end = '')

print(decimalToBinary(24))  

print() 

# Exercise 1.5-1
print("Exercise 1.5-1")

list = [0, 1]

def Fibonacci(num) :
    for i in range(100):
        if list[-1] < num :
            list.append(list[-2]+list[-1])
    
    return list

print(Fibonacci(10))     

print() 

# Exercise 1.5-2
print("Exercise 1.5-2")

digital_0 = ["xxx", "x x", "x x", "x x", "xxx"]
digital_1 = ["  x", "  x", "  x", "  x", "  x"]
digital_2 = ["xxx", "  x", "xxx", "x  ", "xxx"]
digital_3 = ["xxx", "  x", "xxx", "  x", "xxx"]
digital_4 = ["x x", "x x", "xxx", "  x", "  x"]
digital_5 = ["xxx", "x  ", "xxx", "  x", "xxx"]
digital_6 = ["x  ", "x  ", "xxx", "x x", "xxx"]
digital_7 = ["xxx", "x x", "  x", "  x", "  x"]
digital_8 = ["xxx", "x x", "xxx", "x x", "xxx"]
digital_9 = ["xxx", "x x", "xxx", "  x", "  x"]


def display_as_digi(number: int) -> None:
    len_int = len(str(number))
    number_str = str(number)
    for i in range (0, 5):
        for j in number_str:
            if j == "1":
                print(digital_1[i],end="   ")
            elif j == "2":
                print(digital_2[i], end="   ")
            elif j == "3":
                print(digital_3[i], end="   ")
            elif j == "4":
                print(digital_4[i], end="   ")
            elif j == "5":
                print(digital_5[i], end="   ")
            elif j == "6":
                print(digital_6[i], end="   ")
            elif j == "7":
                print(digital_7[i], end="   ")
            elif j == "8":
                print(digital_8[i], end="   ")
            elif j == "9":
                print(digital_9[i], end="   ")
            elif j == "0":
                print(digital_0[i], end="   ")



        print("")
    return

display_as_digi(1234567890)


print() 

# Exercise 2

import numpy as np
x = np.random.random((5,5))
print("Array:")
print(x) 





 


















