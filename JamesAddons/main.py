import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parentdir)
print(os.getcwd())
import main as FEM_main




def FEM_approx():
    FEM_main.main()

if __name__ == '__main__':
    FEM_approx()
    # subprocess.call('./main.py')