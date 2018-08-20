# Author: Donovan Guelde
# CSCI-3104 FAll 2015
# HW 2:  write a python program implementing RSA encryption to encrypt and decrypt a simple message
# On my honor, as a University of Colorado at Boulder student, I have neither given nor received unauthorized assistance.

import random
import math

def isPrime(possiblePrime):
	for k in range(0,3):
		a = random.randint(2,possiblePrime-1)
		if ((a**(possiblePrime-1) % possiblePrime)!=1):
			print possiblePrime, " is not prime"
			return False
		print possiblePrime, " passed ",k," tests for primality where a = ",a
	return True

def numberGenerator(upperLimit):
	possiblePrimeFound = False
	while (possiblePrimeFound == False):
		possiblePrime = random.randint(4,upperLimit) #generates a number between 2 and 2^(bitlimit) - 1
		if (possiblePrime%10 == 1 or possiblePrime%10==3 or possiblePrime%10==7 or possiblePrime%10 ==9):
			possiblePrimeFound = True

	return possiblePrime

def generatePrime(x):
	primeFound = False
	while (primeFound==False):
		possiblePrime = numberGenerator(x)
		if (isPrime(possiblePrime)==True):
			return possiblePrime

def gcd(a,b):
	
	if (b==0):
		return a
	c = a%b
	print c
	return gcd(b, c)

def encrypt(message,e,N):
	return (message**e)%N

def decrypt(encryptedMessage,d,N):
	return pow(encryptedMessage,d)%N

def extendedEuclidean(a,b):
	if (a == 0):
		return b,0,1
	else:
		xprime,yprime,d = extendedEuclidean(b%a,a)
		return (yprime, xprime-(b//a)*yprime, yprime)

def main():
	x = 1
	while (x<2):
		x = input ("Enter bit length for p and q(must be greater than 1): ") #no primes in a 0 or 1 bit-length number
	upperLimit = (1<<(x))-1
	print "upper limit is",upperLimit," 2^",x,"-1"
	p = generatePrime(upperLimit)
	print "p =", p
	q = generatePrime(upperLimit)
	print "q =",q
	N = p*q
	print "n =",N
	phi = (p-1)*(q-1)
	print "phi =",phi
	for e in range (3,phi):
		print "e maybe =",e
		gcdOfE = gcd(e,phi)
		print "gcd =",gcdOfE
		if gcdOfE==1:
			break
	print "e =",e
	xprime,yprime,d = extendedEuclidean(N,e)
	
	message = 7
	encryptedMessage=encrypt(message,e,N)
	decryptedMessage=decrypt(encryptedMessage,d,N)





	print "p =",p," q =",q," N =",N
	print "phi =",phi
	print "e =",e
	print "d =",d
	print "gcd(e,phi) =",gcd(e,phi)
	print "message =",message
	print "encryptedMessage =",encryptedMessage
	print "decryptedMessage =",decryptedMessage



	
	

main()