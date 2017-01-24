// Implement a "good" barrier the "sense reversing" barrier

#include "GoodBarrier.h"

GoodBarrier::GoodBarrier(int P0):
	P(P0), count(P0)
{
	// Initialize the mutex used for FetchAndIncrement
	pthread_mutex_init(&countMutex, 0);
	// Create and initialize the localSense arrar, 1 entry per thread
	localSense = new bool[P];
	for (int i = 0; i < P; ++i) localSense[i] = true;
	// Initialize global sense
	globalSense = true;
}

void GoodBarrier::Enter(int myId)
{ 
	localSense[myId] = !localSense[myId]; // Toggle private sense variable
	if (FetchAndDecrementCount() == 1)
	{ // All threads here, reset count and toggle global sense
		count = P;
		globalSense = localSense[myId];
	}
	else
	{
		while (globalSense != localSense[myId]) { } // Spin
	}
}

int GoodBarrier::FetchAndDecrementCount()
{ // We don’t have an atomic FetchAndDecrement, but we can get the
// same behavior by using a mutex
	pthread_mutex_lock(&countMutex);
	int myCount = count;
	count--;
	pthread_mutex_unlock(&countMutex);
	return myCount;
}
