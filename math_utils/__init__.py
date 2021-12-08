import fractions

from numpy.core.fromnumeric import shape
__version__ = '0.1.0'


class RationalMatrix(object):
    """ 
    Defines a matrix such that every input of the matrix is a rational number. 
    It uses Fraction class from fractions built-in module and of course
    it supports element-wise operations as they defined in Fraction class.
    """
    def __init__(self, matrix):
        self.rmatrix = []
        num_rows = 0
        num_cols = 0

        for row in matrix:
            rrow = []
            for val in row:
                    simplified = fractions.Fraction(val).limit_denominator(10000)
                    rrow.append(simplified)
            self.rmatrix.append(rrow)
            num_rows += 1

            assert num_cols == 0 or num_cols == len(rrow), "matrix cannot empty values"

            num_cols = len(rrow)

        self.shape = (num_rows, num_cols)
        self.T = self.transpose

    def copy(self):
        return RationalMatrix(self.rmatrix)

    def change(self, r1, r2):
        """ Applies R1<->R2 elementary row operation to the matrix. """
        r1, r2 = r1-1, r2-1

        _ = self.rmatrix[r1]
        self.rmatrix[r1] = self.rmatrix[r2]
        self.rmatrix[r2] = _
        return self

    def multiply(self, r1, k):
        """ Multiplies every value in r1-th indexed row with given "k" scalar """
        r1 = r1 - 1
        for idx in range(self.shape[1]):
            self.rmatrix[r1][idx] *= k
        return self        

    def add(self, r1, r2):
        """ Adds r1-th row to r2-th row """
        r1, r2 = r1-1, r2-1
        for idx in range(self.shape[1]):
            self.rmatrix[r2][idx] += self.rmatrix[r1][idx]
        return self

    def transpose(self):
        result = [[0 for i in range(self.shape[0])] for j in range(self.shape[1])]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[j][i] = self.rmatrix[i][j]
        return RationalMatrix(result)


    def __add__(self, obj):
        if isinstance(obj, RationalMatrix):
            assert self.shape == obj.shape, "Matrix shapes must be same for element-wise addition!"
            result = [[0 for i in range(self.shape[1])] for j in range(self.shape[0])]
            for i, row in enumerate(self.rmatrix):
                for j, _ in enumerate(row):
                    result[i][j] = self.rmatrix[i][j] + obj.rmatrix[i][j]
            
            return RationalMatrix(result)

    def __mul__(self, obj):
        if isinstance(obj, (int, float)):
            result = [[0 for i in range(self.shape[1])] for j in range(self.shape[0])]
            for i, row in enumerate(self.rmatrix):
                for j, _ in enumerate(row):
                    result[i][j] = self.rmatrix[i][j] * obj

            return RationalMatrix(result)
        # elif isinstance(obj, RationalMatrix):
        #     t_obj = obj.T()
        #     assert self.shape[1] == obj.shape[0], f"Shapes are incompatible! {self.shape} x {obj.shape}"
        #     result = [[0 for i in range(obj.shape[1])] for j in range(self.shape[0])]
        #     for i, row in enumerate(self.rmatrix):
        #         v = 0
        #         for j, val in enumerate(row):

    def __str__(self):
        str_ = ""
        for row in self.rmatrix:
            str_ += "   ".join(map(lambda fr: f"{fr.numerator}/{fr.denominator}" ,row))
            str_ += "\n"
        return str_

    def __repr__(self):
        return str(self.rmatrix)
    
    def __eq__(self, obj):
        if isinstance(obj, RationalMatrix):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if self.rmatrix[i][j] != obj.rmatrix[i][j]:
                        return False 
            return True

        elif isinstance(obj, list):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if self.rmatrix[i][j] != obj[i][j]:
                        return False 
            return True