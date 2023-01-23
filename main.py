
from sparse import SparseMatrix

def main():
    with open('matrix/input.txt', 'r') as fi:
        with open('matrix/output.txt', 'w') as fo:
            for line in fi:
                matrix = SparseMatrix.parse(line)
                new_matrix = matrix.dot(matrix) + 2*matrix + matrix
                fo.write(str(new_matrix))
                fo.write("\n")

if __name__ == '__main__':
    main()