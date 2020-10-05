# Φ<sub>Flow</sub> QUICK Advection

## Run Test Cases
Go to */custom_apps/test_suit* and execute the command:
```bash
$ python test_suit.py <resolution=100> tf quick
```
The generated fields are then plotted and stored at */custom_apps/test_suit/outputs/quick/*. For a comparison to Semi-Lagrange, execute the command:
```bash
$ python test_suit.py <resolution=100> tf
```
The results are then stored at */custom_apps/test_suit/outputs/semi_lagrange/*. 
It is recommended to first use the standard resolution of 100 cells per axis but the field dimensions can be modified to different values.
```bash
$ python test_suit.py 100 tf quick
```

## Custom Flow with Web-Interface
To execute the Custom App using the QUICK advection scheme, go to */custom_apps/* and run:
```bash
$ python tf_cuda_flow.py tf
```

## Gradient Descent Example
A basic gradient descent scenario is located at */custom_apps/* and can be run by executing:
```bash
$ python tf_cuda_gradients.py tf
```
The results for both QUICK Advection and Semi-Lagrange are stored at */custom_apps/tf_cuda_grad/*.