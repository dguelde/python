# Author: Donovan Guelde
# CSCI-3104 FAll 2015
# HW 2:  write a python program implementing RSA encryption to encrypt and decrypt a simple message
# Automatically executes RSA algorithm with 8,16, and 24 bit-length keys, and displays relevant info

# On my honor, as a University of Colorado at Boulder student, 
# I have neither given nor received unauthorized assistance.

from random import randint
from math import floor
import time

# takes a potentially prime number as input and runs Fermat's Little Theorem test
# to check for primality.  Returns a boolean.
def isPrime(possiblePrime):
	for k in range(0,10): #100 iterations of Fermat's Little Theorem (pretty sure it's gonna be accurate...)
		a = long(randint(2,possiblePrime-1)) #random a between 2 and N-1
		if (modExp(a,possiblePrime-1,possiblePrime)!=1): # a^(N-1) mod N !=1 then not prime
			return False
	return True #passed all tests so true

def numberGenerator(upperLimit): #generates a potentially prime number up to the bit limit specified by user
	possiblePrimeFound = False #flag
	while (possiblePrimeFound == False): #while not found
		possiblePrime = long(randint(2,upperLimit)) #generates a number between 2 and 2^(bitlimit) - 1
		if (possiblePrime%10 == 1 or possiblePrime%10==3 or possiblePrime%10==7 or possiblePrime%10 ==9): 
		#use only if ends in 1,3,7,9
			possiblePrimeFound = True #set flag
	return possiblePrime

def generatePrime(y,x): #generates a prime number between lower and an upper limit
	primeFound = False
	while (primeFound==False):
		possiblePrime = long(numberGenerator(x)) #generate posssibly prime number
		if (isPrime(possiblePrime)==True and possiblePrime>y): #check primality
			return possiblePrime

def gcd(a,b): #finds gcd(a,b) recursively
	if (b==0):
		return a
	return gcd(b, a%b)

def modExp(x,y,n): #returns (x^y) mod n
	if (y==0):
		return 1
	else:
		z = modExp(x,(y//2),n)
		if (y%2==0):
			return ((z*z)%n)
		else:
			return ((x*z**2)%n)

def extendedEuclidean(a,b): #finds gcd(a,b) and returns x,y,d s.t. xa + by = d
	if (b == 0):
		return 1,0,a
	else:
		xprime,yprime,d = extendedEuclidean(b,a % b)
		return (yprime, xprime-(a//b)*yprime, d) #returns x,y,d where xa yN= 1 (mod N)

def generateD(e,N): #generates private key d via extended Euchlid algorithm
	xPrime, yPrime, g = extendedEuclidean(e,N) # xprime*a = 1 mod N (other values discarded)
	if (xPrime<0): #ensure d>0
		xPrime+=N
	return int(xPrime)

def runRSA(bitLength):
	start = start1 = start2 = 0 #reset timer variables 
	message = 2015
	upperLimit = 2**(bitLength-1)-1
	lowerLimit = 2**(bitLength-2) #ensures that the key is 'bitlength' bits (not led by a bunch of zeros)
	start = time.time()
	p = generatePrime(lowerLimit,upperLimit) # p and q are primes in the range (2^bitLimit)-1 and (2^(limit-1))
	q = generatePrime(lowerLimit,upperLimit) #avoids 0000000000000001, etc...
	N = p*q
	phi = (p-1)*(q-1)
	eFound = False
	e = 17
	while (eFound == False):
		gcdEPhi = gcd(e,phi)
		if gcdEPhi==1:
			eFound = True
		else:
			e+=1
	d = generateD(e,phi)
	print "time to generate keys (including p, q, N, phi, d,and e",time.time() - start
	
	encryptedMessage=modExp(message,e,N) #encrypt (message^e)mod N
	decryptedMessage=modExp(encryptedMessage,d,N) #decrypt (encryptedMessage^d)mod N
	#print relevant info
	print "key length:",bitLength,"bits"
	print "p =",p
	print "q =",q
	print "N =  ",N
	print "phi =",phi
	print "message =",message
	start2 = time.time()
	print "encryptedMessage =",encryptedMessage
	print "time to encrypt message",time.time() - start2
	start3 = time.time()
	print "decryptedMessage =",decryptedMessage
	print "time to decrypt message",time.time() - start3
	print "\n"
	return


def main():
	x = input("how many bits")
	runRSA(x)
	#runRSA(16)
	#runRSA(24)
	



	
	

main()