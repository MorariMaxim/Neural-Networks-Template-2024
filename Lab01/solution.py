import re
import math

def parse_equation(equation):
    print(equation)
    matches = re.search(r'([+-]?\d*)x ([+-]) (\d*)y ([+-]) (\d*)z = ([+-]?\d+)', equation)
    
    coefficients = [int(matches.group(1) or '1'), 
                    (1 if matches.group(2) == '+' else -1) * int(matches.group(3) or '1'), 
                    (1 if matches.group(4) == '+' else -1) * int(matches.group(5) or '1')]
    constant = int(matches.group(6))

    print(coefficients,constant)

    return coefficients, constant

def parse_system_of_equations(filename):
    A = []
    B = []
    
    with open(filename, 'r') as file:
        for line in file:
            equation = line.strip()
            if equation:
                coeffs, const = parse_equation(equation)
                A.append(coeffs)
                B.append(const)
    return A,B




def determinant(matrix: list[list[float]]) -> float:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Input must be a non-empty square matrix.")

    if n == 1:
        return matrix[0][0] 

    det = 0.0
    for c in range(n): 
        det += ((-1) ** c) * matrix[0][c] * determinant(minor(matrix,0,c))
    
    return det


def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]
 


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in vector))

def transpose(matrix: list[list[float]]) -> list[list[float]]: 
    return [[matrix[j][i] for j in range(3)] for i in range(3)]

def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]: 
    result = [0.0] * 3
     
    for i in range(3):
        result[i] = sum(matrix[i][j] *vector[j] for j in range(3))

    return result

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]: 
    det_A = determinant(matrix)
    
    if det_A == 0:
        raise ValueError("The system has no unique solution (determinant is zero).")
    
    result = [0.0] * 3
    
    for i in range(3):
        modified_matrix = [row[:] for row in matrix]  
        for j in range(3):
            modified_matrix[j][i] = vector[j]   

        det_Ai = determinant(modified_matrix)

        result[i] = det_Ai / det_A

    return result

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    return [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]

def cofactor(matrix: list[list[float]]) -> list[list[float]]: 
    return [[ (-1)**(i+j) * determinant(minor(matrix,i,j))  for j in range(3)] for i in range(3)]
    
def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))
    

def inverse(matrix: list[list[float]]) -> list[list[float]]: 
    det = determinant(matrix)
    if det == 0:
        raise ValueError("The matrix is singular and cannot be inverted.")
    
    adj = adjoint(matrix)
    return [[adj[i][j] / det for j in range(3)] for i in range(3)]

def solve_inversion(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inv_matrix = inverse(matrix)
    
    result = [0.0] * 3
    for i in range(3):
        result[i] = sum(inv_matrix[i][j] * vector[j] for j in range(3))
    
    return result


if __name__ == "__main__":
    A, B = parse_system_of_equations('system.txt')
    
    print("Matrix A (coefficients):")
    print(A)
    print("\nVector B (constants):")
    print(B)
    
    print(f"{determinant(A)=}")
    
    print(f"{trace(A)=}")
    
    print(f"{norm(B)=}")
    
    print(f"{transpose(A)=}")
    
    print(f"{multiply(A, B)=}")
    
    print(f"{solve_cramer(A, B)=}")
    
    print(f"{solve_inversion(A, B)=}")