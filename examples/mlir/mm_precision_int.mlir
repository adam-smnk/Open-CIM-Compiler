#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>


module {
  func @mm_int16(%arg0: memref<?x?xi16>, %arg1: memref<?x?xi16>, %arg2: memref<?x?xi16>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi16>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi16>) -> index
    %2 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?xi16>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i16, %arg4: i16, %arg5: i16):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i16, i16) -> i16
      %4 = "std.addi"(%3, %arg5) : (i16, i16) -> i16
      "linalg.yield"(%4) : (i16) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} : (memref<?x?xi16>, memref<?x?xi16>, memref<?x?xi16>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_int32(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi32>) -> index
    %2 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %4 = "std.addi"(%3, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_int8(%arg0: memref<?x?xi8>, %arg1: memref<?x?xi8>, %arg2: memref<?x?xi8>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi8>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi8>) -> index
    %2 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?xi8>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i8):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i8, i8) -> i8
      %4 = "std.addi"(%3, %arg5) : (i8, i8) -> i8
      "linalg.yield"(%4) : (i8) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} : (memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi8>) -> ()
    "std.return"() : () -> ()
  }
}