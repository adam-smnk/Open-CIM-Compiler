#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>


module {
  func @tc3x3x2(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    %2 = "std.dim"(%arg0) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    %3 = "std.dim"(%arg1) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %4 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %5 = "std.addi"(%4, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%5) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel"]} : (memref<?x?x?xi32>, memref<?x?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
}
