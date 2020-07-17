# Collection of CIM test examples

Collection of various TC and entry level MLIR examples for testing with CIM dialect.

To convert generate MLIR from TC:
```
path/to/teckyl --emit=mlir ./tc/FILE.tc 2> path/to/FILE.mlir
```

To lower MLIR examples to CIM dialect:
```
path/to/mlir-opt --convert-linalg-to-cim --cse ./mlir/FILE.mlir
```

To lower MLIR examples to Standard dialect:
```
path/to/mlir-opt --convert-linalg-to-cim --convert-cim-to-std --cse ./mlir/FILE.mlir
```

To lower MLIR examples to LLVM dialect:
```
path/to/mlir-opt --convert-linalg-to-cim --convert-cim-to-std --convert-linalg-to-llvm --cse ./mlir/FILE.mlir
```
