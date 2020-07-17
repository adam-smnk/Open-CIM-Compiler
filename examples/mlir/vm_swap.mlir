#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>


module {
  func @vm(%arg0: memref<?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?xi32>) -> index
    %1 = "std.dim"(%arg1) {index = 0 : index} : (memref<?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %2 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %3 = "std.addi"(%2, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map2], iterator_types = ["reduction", "parallel"]} : (memref<?xi32>, memref<?x?xi32>, memref<?xi32>) -> ()
    "std.return"() : () -> ()
  }
}