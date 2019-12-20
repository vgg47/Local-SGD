for worker in range(21):
    print ("mpiexec --oversubscribe -n " + 
        str(worker) + " python3 script.py -s " + 
        str(1) + " -b " + str(1))
    print ("mpiexec --oversubscribe -n " + 
        str(worker) + " python3 script.py -s " + 
        str(1) + " -b " + str(2))
    print ("mpiexec --oversubscribe -n " + 
        str(worker) + " python3 script.py -s " + 
        str(2) + " -b " + str(1))
    print ("mpiexec --oversubscribe -n " + 
        str(worker) + " python3 script.py -s " + 
        str(2) + " -b " + str(2))
