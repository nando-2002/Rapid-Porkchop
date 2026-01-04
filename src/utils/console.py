import os

def printLine():
    columns = os.get_terminal_size().columns
    columns = int(columns / 3)
    for i in range(columns):
        print("-", end = "")
    print("")
