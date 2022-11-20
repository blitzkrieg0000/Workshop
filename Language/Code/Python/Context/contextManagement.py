import os
from contextlib import contextmanager


class Open_File():
    def __init__(self, destination):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        pass

with Open_File("context/files/dump.txt") as f:
    f.write("ABCDE")


@contextmanager
def change_dir(destination):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)

with change_dir('A'):
    print(os.listdir())

with change_dir('B'):
    print(os.listdir())