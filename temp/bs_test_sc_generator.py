for worker in range(1, 21, 2):
    print ("mpiexec --oversubscribe -n " + 
        str(worker) + " python3 script.py --sync " + 
        str(256) + " -b " + str(4))
    print ("mpiexec --oversubscribe -n " + 
        str(worker) + " python3 script.py --sync " + 
        str(256) + " -b " + str(16))
    print ("mpiexec --oversubscribe -n " + 
        str(worker) + " python3 script.py --sync " + 
        str(256) + " -b " + str(32))
    print ("mpiexec --oversubscribe -n " + 
        str(worker) + " python3 script.py --sync " + 
        str(256) + " -b " + str(256))
