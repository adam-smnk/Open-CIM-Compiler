#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d2, d1, d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, d1, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d5)>


module {
  func @mm4x3(%arg0: memref<?x?x?x?xi32>, %arg1: memref<?x?x?x?x?xi32>, %arg2: memref<?x?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?x?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?x?x?xi32>) -> index
    %2 = "std.dim"(%arg0) {index = 2 : index} : (memref<?x?x?x?xi32>) -> index
    %3 = "std.dim"(%arg0) {index = 3 : index} : (memref<?x?x?x?xi32>) -> index
    %4 = "std.dim"(%arg1) {index = 0 : index} : (memref<?x?x?x?x?xi32>) -> index
    %5 = "std.dim"(%arg1) {index = 4 : index} : (memref<?x?x?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %6 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %7 = "std.addi"(%6, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%7) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction", "parallel"]} : (memref<?x?x?x?xi32>, memref<?x?x?x?x?xi32>, memref<?x?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
}