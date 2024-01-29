"""
Module to Find CpGs in The read:
Algorithm: 
ord("C") + ord("G") = 138
From left to right, check the sum of every to consecutive numbers, if the number == 138
accumate one to CpG counter
at the end, divide the number by the read length, accumulate 1 to the CpG Tensor at the end of the rounded number.
"""
