import os

def printLine():
    columns = os.get_terminal_size().columns
    for i in range(columns):
        print("-", end = "")

printLine()