class GoodBarrier {
	public: 
		GoodBarrier(int P0); // P is the total number of threads
		void Enter(int myId); // Enter the barrier, don’t exit till alll there
	private:
		int P;
		int count; // Number of threads presently in the barrier
		int FetchAndDecrementCount();
		pthread_mutex_t countMutex;
		bool* localSense; // We will create an array of bools, one per thread
		bool globalSense; // Global sense
};
