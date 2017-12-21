# all the computations for these problems are done using binary arithmetic
# only the user input and the final output will be in decimal.
# dec2bin and bin2dec convert between binary and decimal.


import random
import sys
import time

sys.setrecursionlimit(10000000)

from random import *

def Problem1Proj2(N, k):
    #simply consumes the primality3 function to check primality
    if primality3(dec2bin(N), k):
        print("N is a prime")
    else:
        print("N is not a prime")

def Problem2Proj2(N, k):
    #Mainly utilizes the genPrime function and then prints that.
    bitVec = genPrime(N, k)
    print("Integer %s is a prime" %bin2dec(bitVec))

def Problem3Proj2(n, k):
    #Calculate time to generate all required values
    start_time = time.time()
    #calculate E, D, N, then encrypt and decrypt message M.
    #create two empty bit vectors
    p = []
    q = []
    #keep finding a prime number until they are both different primes
    while compare(p, q) == 0:
        p = genPrime(n, k)
        q = genPrime(n, k)
    #calculate N and generate a random E
    N = mult(p, q)
    E = randomBitVec(k)
    #make E a new bitvector until gcd(E, (p-1)*(q*1)) = 1 (coprimes)
    while bin2dec(gcd(E, mult(sub(p, dec2bin(1)), sub(q, dec2bin(1))))) != 1:
        E = randomBitVec(k)
    #find D through the modinverse function
    D = modinv(E, mult(sub(p, dec2bin(1)), sub(q, dec2bin(1))))
    print("--- %s Seconds to generate cryptographic values ---" %(time.time() - start_time))
    print("N: %s" %bin2dec(N))
    print("E: %s" %bin2dec(E))
    print("D: %s" %bin2dec(D))
    M = int(input("Please enter a message as an integer: "))
    #calculate Cipher by raising binary message M to power E mod N
    C = modexp(dec2bin(M), E, N)
    print("Encrypted Message: %s" %bin2dec(C))
    #decryption statement to get CPrime (the decrypted Message)
    CPrime = modexp(C, D, N)
    print("Decrypted Message: %s" %bin2dec(CPrime))
    
def Problem1(A, B, C, D):
    #This problem calculates A^B - C^D
    A1 = dec2bin(A)
    C1 = dec2bin(C)
    print (bin2dec(sub(exp(A1, B), exp(C1, D))))

def Problem2(A, B, C, D):
    #this problem calculates A^B / C^D
    A1 = dec2bin(A)
    C1 = dec2bin(C)
    (q, r) = (divide(exp(A1, B), exp(C1, D)))
    print ("quotient:")
    print (bin2dec(q))
    print ("remainder:")
    print (bin2dec(r))

def Problem3(A):
    #this problem calculates sum of 1/1 + 1/2 +... + 1/A
    (n, d) = problem3Help(dec2bin(1), dec2bin(A))
    G = gcd(n, d)
    print ("Numerator:")
    (q, r) = divide(n, G) 
    print (bin2dec(q))
    print ("Denominator:")
    (q, r) = divide(d, G)
    print (bin2dec(q))
    
def problem3Help(A, B):
    #recursively goes through to calculate totals for numerator and denominator
    if compare(B, dec2bin(1)) == 0:
        return (B,A)
    #incrementing A by 1 for the depth of the series
    (n, d) = problem3Help(add(A, dec2bin(1)), sub(B, dec2bin(1)))
    #multiply A by the bottom of other factor i.e. 1/2 + 1/3 == 3/6 + 2/6 
    return (add(mult(n, A), d), mult(d, A))  

def primality(N):
    #generate random integer 1 < X < N
    X = randint(2, bin2dec(sub(N, dec2bin(1))))
    #call modular exponentiation function to check x^(N - 1) = 1 mod N
    r = modexp(dec2bin(X), sub(N, dec2bin(1)), N)
    
    #1 is the good sign!
    if bin2dec(r) == 1:
        return True
    else :
        return False

def primality2(N, k):
    #simply loop in range of confidence to check primality
    for i in range (0, k - 1):
        if not primality(N):
            return False
    return True
def primality3(N, k):
    #check primality by common Divisors: i.e. remainder = 0
    (q, r) = divide(N, dec2bin(2))
    if bin2dec(r) == 0:
        return False
    (q, r) = divide(N, dec2bin(3))
    if bin2dec(r) == 0:
        return False
    (q, r) = divide(N, dec2bin(5))
    if bin2dec(r) == 0:
        return False
    (q, r) = divide(N, dec2bin(7))
    if bin2dec(r) == 0:
        return False
    #Call primality2 to run loop.
    if primality2(N, k):
        return True
    return False

def genPrime(N, k):
    #keep generating random bitvectors until it is prime with k confidence
    while True:
        bitVec = randomBitVec(N)
        if primality3(bitVec, k):
            return bitVec

def randomBitVec(N):
    #create empty bit vector
    bitVec = []
    #fill vector randomly with 1's and 0's
    for i in range(0, N - 2):
        bitVec.append(randint(0, 1))
    #add 1 to front and back of the bit vector
    bitVec.append(1)
    bitVec.insert(0, 1)
    return bitVec
    
def modinv(A, B):
    #find the extendedEuclid
    (x, y) = extendedEuclid(A, B)
    #if the negative flag is set
    if x[len(x) - 1] == -1:
        #we delete the negative flag and increment x by B:
        #example -117 with N of 352 -> 352 - 117 = 235 (the correct D)
        del x[len(x) - 1]
        x = sub(B, x)
    return x

def extendedEuclid(A, B):
    if zero(B):
        #set D and return 1 and 0 binary
        return (dec2bin(1), dec2bin(0))
    #get the remainder and the q, ones used in recursion, the other in the return
    (q, r) = divide(A, B)
    (x, y) = extendedEuclid(B, r)
    #just reset x and y if they're 0, sometimes get get trimmed to no items yikes!
    if len(x) == 0:
        x = [0]
    if len(y) == 0:
        y = [0]
    if len(y) > 0:
        #if y is negative
        if y[len(y) - 1] == -1:
            #create a new non negative version of y that we can math with
            retY = y[0:len(y)-1]
            if len(x) > 0:
                #if x is negative
                if x[len(x) - 1] == -1:
                    #make x a non negative, since X - QY and x and y < 0, -QY > 0, so QY - X suffices
                    del x[len(x) - 1]
                    return (y, sub(mult(q, retY), x))
            #just y < 0 -> -QY > 0 -> X + QY
            return (y, add(mult(q, retY), x))
    if len(x) > 0:
        #check if x is negative
        if x[len(x) - 1] == -1:
            #This means X < 0, so make non negative, X - QY = -(|X|+QY)
            del x[len(x) - 1]
            a = add(x, mult(q, y))
            a.append(-1)
            return (y, a)
    #Simply use the standard case, neither are negative.
    return (y, sub(x, mult(q, y)))

def modexp(x, y, N):
    #this is really just the modular exponentiation algorithm
    if bin2dec(y) == 0:
        return [1]
    (q, r) = divide(y, dec2bin(2))
    z = modexp(x, q, N)
    if even(y):
        (q, r) = divide(exp(z, 2), N)
        return r
    else:
        (q, r) = divide(mult(x, exp(z, 2)), N)
        return r

def shift(A, n):
    if n == 0:
        return A
    return [0 ] +shift(A, n-1)


def mult(X, Y):
    # mutiplies two arrays of binary numbers
    # with LSB stored in index 0
    if zero(Y):
        return [0]
    Z = mult(X, div2(Y))
    if even(Y):
        return add(Z, Z)
    else:
        return add(X, add(Z, Z))


def Mult(X, Y):
    X1 = dec2bin(X)
    Y1 = dec2bin(Y)
    return bin2dec(mult(X1, Y1))


def zero(X):
    # test if the input binary number is 0
    # we use both [] and [0, 0, ..., 0] to represent 0
    if len(X) == 0:
        return True
    else:
        for j in range(len(X)):
            if X[j] == 1:
                return False
    return True


def div2(Y):
    if len(Y) == 0:
        return Y
    else:
        return Y[1:]


def even(X):
    if ((len(X) == 0) or (X[0] == 0)):
        return True
    else:
        return False
#################################################
#               Addition Functions              #
#################################################


def add(A, B):
    A1 = A[:]
    B1 = B[:]
    n = len(A1)
    m = len(B1)
    if n < m:
        for j in range(len(B1) - len(A1)):
            A1.append(0) #This adds to the A List
    else:
        for j in range(len(A1) - len(B1)):
            B1.append(0) #This adds to the B1 List
    N = max(m, n)
    C = []
    carry = int(0)
    for j in range(N):
        C.append(exc_or(int(A1[j]), int(B1[j]), int(carry)))
        carry = nextcarry(int(carry), int(A1[j]), int(B1[j]))
    if carry == 1:
        C.append(carry)
    return C


def Add(A, B):
    return bin2dec(add(dec2bin(A), dec2bin(B)))
#################################################
#               Subtraction Functions           #
#################################################
def sub(X,Y):
    A1 = X[:]
    B1 = Y[:]
    n = len(A1)
    m = len(B1)
    negative = False
    if zero(Y):
        return X
    if n < m:
        for j in range(len(B1) - len(A1)):
            A1.append(0)  # This adds to the A List
    else:
        for j in range(len(A1) - len(B1)):
            B1.append(0)  # This adds to the B1 List
    A1.append(0)
    B1.append(0)
    for j in range(len(B1)):
        if B1[j] == 1:
            B1[j] = 0
        else:
            B1[j] = 1
    BC = add(dec2bin(1), B1)
    S = add(A1, BC)
    if len(S) > len(BC):
        S.pop()
    if S[len(S) - 1] == 1:
        negative = True
        for j in range(len(S)):
            if S[j] == 1:
                S[j] = 0
            else:
                S[j] = 1
        S = add(dec2bin(1), S)
        
    S.pop()
    if negative:
        S.append(-1)
    return S

def Sub(A,B):
    return bin2dec(sub(dec2bin(A), dec2bin(B)))

def exp(A,B):
    A1 = A[:]
    tot = A[:]
    for j in range(B-1):
        tot = mult(A1, tot)
    return tot

def Exp(A,B):
    return bin2dec(exp(dec2bin(A), B))


def exc_or(a, b, c):
    return (a ^ (b ^ c))


def nextcarry(a, b, c):
    if ((a & b) | (b & c) | (c & a)):
        return 1
    else:
        return 0


def bin2dec(A):
    if len(A) == 0:
        return 0
    multiple = 1
    if A[len(A) - 1] == -1:
        del A[len(A) - 1]
        multiple = -1
    val = A[0]
    pow = 2
    for j in range(1, len(A)):
        val = val + pow * A[j]
        pow = pow * 2
    return val * multiple


def reverse(A):
    B = A[::-1]
    return B


def trim(A):
    if len(A) == 0:
        return A
    A1 = reverse(A)
    while ((not (len(A1) == 0)) and (A1[0] == 0)):
        A1.pop(0)
    return reverse(A1)


def compare(A, B):
    # compares A and B outputs 1 if A > B, 2 if B > A and 0 if A == B
    A1 = reverse(trim(A))
    A2 = reverse(trim(B))
    if len(A1) > len(A2):
        return 1
    elif len(A1) < len(A2):
        return 2
    else:
        for j in range(len(A1)):
            if A1[j] > A2[j]:
                return 1
            elif A1[j] < A2[j]:
                return 2
        return 0


def Compare(A, B):
    return bin2dec(compare(dec2bin(A), dec2bin(B)))


def dec2bin(n):
    if n == 0:
        return []
    m = n / 2
    A = dec2bin(m)
    fbit = n % 2
    return [fbit] + A


def map(v):
    if v == []:
        return '0'
    elif v == [0]:
        return '0'
    elif v == [1]:
        return '1'
    elif v == [0, 1]:
        return '2'
    elif v == [1, 1]:
        return '3'
    elif v == [0, 0, 1]:
        return '4'
    elif v == [1, 0, 1]:
        return '5'
    elif v == [0, 1, 1]:
        return '6'
    elif v == [1, 1, 1]:
        return '7'
    elif v == [0, 0, 0, 1]:
        return '8'
    elif v == [1, 0, 0, 1]:
        return '9'


def bin2dec1(n):
    if len(n) <= 3:
        return map(n)
    else:
        temp1, temp2 = divide(n, [0, 1, 0, 1])
        return bin2dec1(trim(temp1)) + map(trim(temp2))


def divide(X, Y):
    # finds quotient and remainder when X is divided by Y
    if zero(X):
        return ([], [])
    (q, r) = divide(div2(X), Y)
    q = add(q, q)
    r = add(r, r)
    if not even(X):
        r = add(r, [1])
    if not compare(r, Y) == 2:
        r = sub(r, Y)
        q = add(q, [1])
    return (q, r)


def Divide(X, Y):
    (q, r) = divide(dec2bin(X), dec2bin(Y))
    return (bin2dec(q), bin2dec(r))

def gcd(A,B):
    if not zero(B):
        q, r = divide(A, B)
        return gcd(B, r)
    else:
        return A

def GCD(A,B):
    return bin2dec(gcd(dec2bin(A), dec2bin(B)))


def main():
    I = int(input("Select a function: \n1. A^B - C^D\n2. A^B / C^D\n3. 1/1 + ... + 1/n\n4. Primality Test\n5. Generate N bit prime\n6. Encrypt and Decrypt\nOr 7 to exit\n"))
    while I != 7:
        if I == 1:
            print("inside selection")
            print("Selection: A^B - C^D:")
            A = int(input("Enter an A value:\n"))
            B = int(input("Enter an B value:\n"))
            C = int(input("Enter an C value:\n"))
            D = int(input("Enter an D value:\n"))
            Problem1(A, B, C, D)
        if I == 2:
            print("Selection: A^B / C^D:")
            A = int(input("Enter an A value:\n"))
            B = int(input("Enter an B value:\n"))
            C = int(input("Enter an C value:\n"))
            D = int(input("Enter an D value:\n"))
            Problem2(A, B, C, D)
        if I == 3:
            print("Selection: 1/1 + ... + 1/n")
            A = int(input("Enter an A value: \n"))
            Problem3(A)
        if I == 4:
            print("Selection: Primality test")
            A = int(input("Enter a possible prime N: "))
            k = int(input("Enter a confidence k: "))
            Problem1Proj2(A, k) 
        if I == 5:
            print("Selection: Generate N bit prime")
            A = int(input("Enter a bit length N: "))
            k = int(input("Enter a confidence k: "))
            Problem2Proj2(A, k)
        if I == 6:
            print("Selection: Encrypt and Decrypt")
            A = int(input("Enter a bit length N: "))
            k = int(input("Enter a confidence k: "))
            Problem3Proj2(A, k)
        I = int(input("Select another function: \n1. A^B - C^D\n2. A^B / C^D\n3. 1/1 + ... + 1/n\n4. Primality Test\n5. Generate N bit prime\n6. Encrypt and Decrypt\nOr 7 to exit\n"))
        
if __name__ == '__main__':
    main()








