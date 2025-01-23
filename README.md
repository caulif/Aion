# Aion

Aion is a system built for privacy-preserving federated learning, where individual training weights are aggregated using secure aggregation. 

## Overview
We integrate our code into [ABIDES](https://github.com/jpmorganchase/abides-jpmc-public), an open-source highfidelity simulator designed for AI research in financial markets (e.g., stock exchanges). 
The simulator supports tens of thousands of clients interacting with a server to facilitate transactions (and in our case to compute sums). 
It also supports configurable pairwise network latencies.

Aion protocol works by steps (i.e., round trips). 
A step includes waiting and processing messages. 
The waiting time is set according to the network latency distribution and a target dropout 



## Installation Instructions




## **Secure Aggregation**

First enter into folder `pki_files`, and run
```
python setup_pki.py
```

Our program has multiple configs.

```

```
Aion supports batches of clients with size power of 2, starting from 128,
e.g., 128, 256, 512.

Example command:
```
python abides.py -c Aion -n 128 -i 1 -p 1 
```



## Test of Defend Against Poisoning Attacks


Example command: 
```

```



## Additional Information

The server waiting time is set in `util/param.py` according to a target dropout rate (1%).
Specifically, for a target dropout rate, we set the waiting time according to the network latency (see `model/LatencyModel.py`). For each iteration, server total time = server waiting time + server computation time.

