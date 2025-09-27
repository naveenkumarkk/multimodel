import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

def abspath(*paths):
    return os.path.join(BASE_DIR, *paths)
