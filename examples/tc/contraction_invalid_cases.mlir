#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d0, d1, d4)>
#map4 = affine_map<(d0, d1, d2) -> (d0)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2) -> (d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map9 = affine_map<(d0) -> (d0)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map11 = affine_map<(d0, d1) -> (d0, d1)>
#map12 = affine_map<(d0, d1) -> (d1, d0)>
#map13 = affine_map<(d0, d1) -> (d0)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>


module {
  func @mm_dim_err(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    %2 = "std.dim"(%arg0) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    %3 = "std.dim"(%arg1) {index = 0 : index} : (memref<?x?x?xi32>) -> index
    %4 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    %5 = "std.dim"(%arg1) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %6 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %7 = "std.addi"(%6, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%7) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction", "reduction"]} : (memref<?x?x?xi32>, memref<?x?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err1(%arg0: memref<?x?xi32>, %arg1: memref<?x?x?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi32>) -> index
    %2 = "std.dim"(%arg1) {index = 0 : index} : (memref<?x?x?x?xi32>) -> index
    %3 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?x?x?xi32>) -> index
    %4 = "std.dim"(%arg1) {index = 2 : index} : (memref<?x?x?x?xi32>) -> index
    %5 = "std.dim"(%arg1) {index = 3 : index} : (memref<?x?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %6 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %7 = "std.addi"(%6, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%7) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map2, #map3, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction", "reduction"]} : (memref<?x?xi32>, memref<?x?x?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err2(%arg0: memref<?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?xi32>) -> index
    %1 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?xi32>) -> index
    %2 = "std.dim"(%arg2) {index = 1 : index} : (memref<?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %4 = "std.addi"(%3, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} : (memref<?xi32>, memref<?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err3(%arg0: memref<?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?xi32>) -> index
    %1 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?xi32>) -> index
    %2 = "std.dim"(%arg2) {index = 0 : index} : (memref<?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %4 = "std.addi"(%3, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map4, #map5, #map7], iterator_types = ["reduction", "parallel", "reduction"]} : (memref<?xi32>, memref<?x?xi32>, memref<?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err4(%arg0: memref<?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?xi32>) -> index
    %1 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    %2 = "std.dim"(%arg1) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %4 = "std.addi"(%3, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map4, #map8, #map7], iterator_types = ["reduction", "parallel", "reduction"]} : (memref<?xi32>, memref<?x?x?xi32>, memref<?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err5(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %1 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %2 = "std.addi"(%1, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%2) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} : (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err6(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?x?xi32>) -> index
    %2 = "std.dim"(%arg0) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %3 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %4 = "std.addi"(%3, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map10, #map10, #map10], iterator_types = ["parallel", "parallel", "parallel"]} : (memref<?x?x?xi32>, memref<?x?x?xi32>, memref<?x?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err7(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %2 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %3 = "std.addi"(%2, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map11, #map11, #map11], iterator_types = ["parallel", "parallel"]} : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err8(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %2 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %3 = "std.addi"(%2, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map11, #map12, #map11], iterator_types = ["parallel", "parallel"]} : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err9(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %2 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %3 = "std.addi"(%2, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "reduction"]} : (memref<?x?xi32>, memref<?x?xi32>, memref<?xi32>) -> ()
    "std.return"() : () -> ()
  }
  func @mm_dim_err_a1(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?x?xi32>) {
    %0 = "std.dim"(%arg0) {index = 0 : index} : (memref<?x?xi32>) -> index
    %1 = "std.dim"(%arg0) {index = 1 : index} : (memref<?x?xi32>) -> index
    %2 = "std.dim"(%arg1) {index = 1 : index} : (memref<?x?xi32>) -> index
    %3 = "std.dim"(%arg2) {index = 2 : index} : (memref<?x?x?xi32>) -> index
    "linalg.generic"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
      %4 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
      %5 = "std.addi"(%4, %arg5) : (i32, i32) -> i32
      "linalg.yield"(%5) : (i32) -> ()
    }) {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map14, #map15, #map16], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?x?xi32>) -> ()
    "std.return"() : () -> ()
  }
}