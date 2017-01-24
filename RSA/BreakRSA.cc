// Grad student portion of the RSA assignment
// Fall 2015 ECE6122

#include <iostream>

#include "RSA_Algorithm.h"
#include <math.h>
#include <stdint.h>
#define numPerArg 6

using namespace std;

mpz_class gcd( mpz_class a, mpz_class b) {
	mpz_class remainder;
	while (b != 0) {
		remainder = a % b;
		a = b;
		b = remainder;
	}
	return a;
}

int main(int argc, char** argv)
{ // Arguments are as follows:
  //argv[1] = n;
  //argv[2] = e;  // n and e are the public key
  //argv[3] = first 6 characters of encrypted message
  //argv[4] = next 6 characters .. up to argv[12] are the lsat 6 characters
  // The number of arguments will always be exacty 12, and each argument past the
  // public key contain 6 ascii characters of the encrypted message.
  // Each of the 32bit values in the argv[] array are right justified in the
  // low order 48 bits of each unsigned long.  The upper 16 bits are always
  // zero, which insures the each unsigned long is < n (64 bits) and therefore
  // the RSA encryption will work.

  // Below is an example of the BreakRSA and command line arguments:

// ./BreakRSA  2966772883822367927 2642027824495698257  817537070500556663 1328829247235192134 1451942276855579785 2150743175814047358 72488230455769594 1989174916172335943 962538406513796755 1069665942590443121 72678741742252898 1379869649761557209

//   The corect output from the above is:
//   HelloTest  riley CekwkABRIZFlqmWTanyXLogFgBUENvzwHpEHRCZIKRZ
//
//   The broken (computed) private key for the above is 4105243553



  // Our one and only RSA_Algorithm object
  RSA_Algorithm rsa;
  mpz_class p, q, phi;
  // First "break" the keys by factoring n and computing d
  // Set the keys in the rsa object afer calculating d
  rsa.n = mpz_class(argv[1]);
  rsa.e = mpz_class(argv[2]);
  
  //Factor n using Pollard Rho algorithm
  mpz_class x_fixed = 2,cycle_size = 2,x = 2, temp;
  p = 1;

	while (p == 1) {

		for (mpz_class count=1;count <= cycle_size && p<= 1;count++) {
			x = (x*x+1);
			mpz_mod(x.get_mpz_t(), x.get_mpz_t(), rsa.n.get_mpz_t());
			temp = x-x_fixed;
			mpz_gcd (p.get_mpz_t(), temp.get_mpz_t(), rsa.n.get_mpz_t());
		}

		cycle_size *= 2;
		x_fixed = x;
	}

  q = rsa.n/p;
  //cout << "computed n = "<<p*q<<endl;
  phi = (p-1)*(q-1);

  // Set rsa.d to the calculated private key d
  mpz_invert(rsa.d.get_mpz_t(), rsa.e.get_mpz_t(), phi.get_mpz_t());
  //cout<< "d=" <<rsa.d<<endl;
  
  for (int i = 3; i < 13; ++i)
    { // Decrypt each set of 6 characters
      mpz_class c(argv[i]);
      mpz_class m = rsa.Decrypt(c);
      //cout<<"m = " <<m<<endl;
      //  use the get_ui() method in mpz_class to get the lower 48 bits of the m
      unsigned long ul = m.get_ui();
      //cout << ul << endl;
      
      // Now print the 6 ascii values in variable ul.  As stated above the 6 characters are in the low order 48 bits of ui.
      for(int value = 5; value > -1; --value)
      {
		uint64_t shifted = ul >> (value*8);
		shifted = shifted & 0xFF;
		//cout <<"\n shiefted= "<<shifted<< " \t";
		cout<< (char)shifted;
  	  }
      
	}
}

