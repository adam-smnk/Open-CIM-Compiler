#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d0)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d4, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4, d1, d0, d3)>
#map6 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d2, d0)>


module {
  func @mm(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?x?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    %2 = "std.dim"(%arg0) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    %3 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    %4 = "std.dim"(%arg1) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %5 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %6 = "std.addi"(%5, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%6) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} : (memref<?x?x?xi32>, memref<?x?x?xi32>, memref<?x?x?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm2(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?x?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    %2 = "std.dim"(%arg0) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    %3 = "std.dim"(%arg1) {index = 0 : index} : (memref<?x?x?xi32>) -> index
    %4 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %5 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %6 = "std.addi"(%5, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%6) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} : (memref<?x?x?xi32>, memref<?x?x?xi32>, memref<?x?x?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm3(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi32>) -> index
    %2 = "std.dim"(%arg1) {index = 0 : index} : (memref<?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %4 = "std.addi"(%3, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm4(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi32>) -> index
    %2 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %4 = "std.addi"(%3, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map6, #map9, #map8], iterator_types = ["parallel", "parallel", "reduction"]} : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
}