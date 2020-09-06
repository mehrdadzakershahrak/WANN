import pickle
import os

def main():
    with open(f'result{os.sep}_checkpoint{os.sep}run-checkpoint.pkl', 'rb') as f:
        test = pickle.load(f)

    print('test')


if __name__ == '__main__':
    main()
