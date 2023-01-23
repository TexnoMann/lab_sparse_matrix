from __future__ import annotations
from typing import Tuple
import ast
import numpy as np
import copy


class SparseMatrix:
    """
    A class that stores a sparse matrix in CSR format.
    """
    def __init__(self,
        values: tuple,
        cols_indices: tuple,
        start_rows_indices: tuple,
        shape: Tuple[int, int] = None
    ): 
        self.__values = values
        self.__cols_indices = cols_indices
        self.__start_rows_indices= start_rows_indices

        self.__nnz = len(values)
        assert len(self.__start_rows_indices) >= 1, "The size of start_rows_indeces argument must be >= 1"
        if shape is None:
            self.__shape = (len(self.__start_rows_indices)-1, max(self.__cols_indices))
        else:
            self.__shape = shape
        assert len(self.__shape)==2, "Given shape of matrix is invalid. It must be tuple with lenght =2"
    
    def toarray(self) -> np.ndarray:
        output_matrix = np.zeros(self.__shape)
        for ri in range(0, len(self.__start_rows_indices)-1):
            non_zero_in_row_elements_indices = list(
                range(self.__start_rows_indices[ri], self.__start_rows_indices[ri+1])
            )
            for ind in non_zero_in_row_elements_indices:
                output_matrix[ri, self.__cols_indices[ind]] = self.__values[ind]
        return output_matrix
    
    def __mul__(self, number: float) -> SparseMatrix:
        if not isinstance(number, (float, int)):
            raise ValueError('Unknown argument for multiplication. There is not a number. For matrix multiplication use dot()')
        new_matrix = copy.deepcopy(self)
        new_matrix.values = [number*v for v in self.values]
        return new_matrix

    def __rmul__(self, number: float) -> SparseMatrix:
        return self.__mul__(number)
    
    def dot(self, matrix: SparseMatrix):
        """
        There is a implementation of row-wise matrix product
        """
        new_values = []
        new_cols_indices = []
        new_start_row_indices = []

        assert matrix.shape[0] == self.__shape[1], "Incompatible matrix shapes"

        for ri in range(0, self.__shape[0]):
            # Getting needed right matrix rows that matching for nonzero values in row of the left matrix 
            non_zero_in_row_elements_indices = list(
                range(self.__start_rows_indices[ri], self.__start_rows_indices[ri+1])
            )

            output_matrix_value_by_column = {}
            
            # Iterate by right matrix rows
            for nze in non_zero_in_row_elements_indices:
                nrow = self.__cols_indices[nze]
                right_non_zero_in_row_elements_indices = list(
                    range(matrix.start_rows_indices[nrow], matrix.start_rows_indices[nrow+1])
                )
                for ind in right_non_zero_in_row_elements_indices:
                    if matrix.cols_indices[ind] in output_matrix_value_by_column:
                        output_matrix_value_by_column[matrix.cols_indices[ind]] += matrix.values[ind]*self.__values[nze]
                    else:
                        output_matrix_value_by_column[matrix.cols_indices[ind]] = matrix.values[ind]*self.__values[nze]

            new_start_row_indices.append(len(new_values))
            # Sort column index -> value map by column number
            for k in sorted(output_matrix_value_by_column, key=output_matrix_value_by_column.get, reverse=False):
                new_values.append(output_matrix_value_by_column[k])
                new_cols_indices.append(k)
            
        new_start_row_indices.append(len(new_values))
            
        return SparseMatrix(new_values, new_cols_indices, new_start_row_indices, shape=(self.shape[0], matrix.shape[1]))
    
    def __str__(self):
        return str(self.toarray().tolist())
    
    def __add__(self, matrix: SparseMatrix):
        """
        There is a implementation of row-wise matrix sum
        """
        new_values = []
        new_cols_indices = []
        new_start_row_indices = []

        assert matrix.shape[0] == self.__shape[0] and matrix.shape[1] == self.__shape[1], "Incompatible matrix shapes"

        for ri in range(0, self.__shape[0]):
            # Getting needed right matrix rows that matching for nonzero values in row of the left matrix 
            non_zero_in_row_elements_indices = list(
                range(self.__start_rows_indices[ri], self.__start_rows_indices[ri+1])
            )
            right_non_zero_in_row_elements_indices = list(
                range(matrix.start_rows_indices[ri], matrix.start_rows_indices[ri+1])
            )

            output_matrix_value_by_column = {}
            
            # Iterate by right matrix rows
            for nze in non_zero_in_row_elements_indices:
                ncol = self.__cols_indices[nze]
                output_matrix_value_by_column[ncol] = self.__values[nze]
            
            for nze in right_non_zero_in_row_elements_indices:
                ncol =matrix.cols_indices[nze]
                if ncol in output_matrix_value_by_column:
                    output_matrix_value_by_column[ncol] += matrix.values[nze]
                else:
                    output_matrix_value_by_column[ncol] = matrix.values[nze]
            
            new_start_row_indices.append(len(new_values))
            # Sort column index -> value map by column number
            for k in sorted(output_matrix_value_by_column, key=output_matrix_value_by_column.get, reverse=False):
                new_values.append(output_matrix_value_by_column[k])
                new_cols_indices.append(k)
            
        new_start_row_indices.append(len(new_values))

        return SparseMatrix(new_values, new_cols_indices, new_start_row_indices, shape=copy.copy(self.shape))        

    @property
    def shape(self) -> Tuple[int, int]:
        return self.__shape

    @property
    def nnz(self) -> int:
        return self.__nnz
    
    @property
    def values(self) -> list:
        return self.__values
    
    @values.setter
    def values(self, values: list):
        self.__values = values
    
    @property
    def cols_indices(self) -> list:
        return self.__cols_indices
    
    @property
    def start_rows_indices(self) -> list:
        return self.__start_rows_indices
    
    @staticmethod
    def parse(literal_matrix: str) -> SparseMatrix:
        """
        Converting string literal form of sparse matrix to
        instance of the SparseMatrix
        """
        try:
            matrix_array = np.array(ast.literal_eval(literal_matrix))
        except ValueError:
            raise ValueError('String cannot be parsed to matrix')
        return SparseMatrix.from_array(matrix_array)
    

    @staticmethod
    def from_array(array: np.ndarray) -> SparseMatrix:
        values = []
        cols_indices = []
        start_rows_indices = []
        for i in range(0, array.shape[0]):
            # First in row nonzero index of element in value array
            start_rows_indices.append(len(values))
            for j in range(0, array.shape[1]):
                element = array[i,j]
                if element != 0.0:
                    values.append(element)
                    cols_indices.append(j)
        start_rows_indices.append(len(values))

        return SparseMatrix(
            values, 
            cols_indices, 
            start_rows_indices, 
            shape=array.shape
        )