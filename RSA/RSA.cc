// ECE4122/6122 RSA Encryption/Decryption assignment
// Fall Semester 2015

#include <iostream>
#include "RSA_Algorithm.h"
#include <time.h>


using namespace std;

// debug
int good = 0;
int bad  =0;
mpz_class e;
mpz_class d;
mpz_class n;
//mpz_class num = 10;
//mpz_class *array = new mpz_class[num.get_ui()];

mpz_class M, C, AD;
//mpz_init2(plaintext, sz * 2);
		
int main()
{
  // Instantiate the one and only RSA_Algorithm object
  RSA_Algorithm RSA;
    
  // Loop from sz = 32 to 1024 inclusive
	for (int sz= 32; sz<=1024; sz*=2)
	{
		cout<<"bit size = " << sz <<endl;
		// for each size choose 10 different key pairs
		for(int key=0; key<10 ; key++){
			while(true){
				int e_found = RSA.GenerateRandomKeyPair(sz);
				if (e_found) { 
					RSA.PrintNDE();
					break; }
			}
// For each key pair choose 10 differnt plaintext messages making sure it is smaller than n.
/*for (mpz_class message=0; message < num.get_ui(); message++){
array[message.get_ui()] = RSA.GetRandom(2*sz-1);
// If not smaller then n then choose another
if(array[message.get_ui()] > n) --message;
}*/

		
		// For each key pair choose 10 differnt plaintext messages making sure it is smaller than n.
		for (int message=0; message<10; message++){
			M = RSA.GetRandom(sz);
			RSA.PrintM(M);
		
		// After encryption, decrypt the ciphertext and verify it matches
			C = RSA.Encrypt(M);
			RSA.PrintC ( C );
			AD = RSA.Decrypt(C);
		//if (AD  == M ) cout << "SUCCESS!"<<endl;
		}
		}
	}
}
