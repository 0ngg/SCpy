
SetFactory("OpenCASCADE");
Point(1) = {-1, 0.5, 0, 1.0};
Point(2) = {-1, -0.5, 0 ,1.0};
Point(3) = {1, -0.5, 0, 1.0};
Point(4) = {1, 0.5, 0, 1.0};

Point(5) = {-1, 0.4, 0, 1.0};
Point(6) = {1, 0.4, 0, 1.0};
Point(7) = {-1, -0.45, 0, 1.0};
Point(8) = {1, -0.45, 0, 1.0};
Line(10) = {5, 6};
Line(11) = {7, 8};
Line(12) = {1, 5};
Line(13) = {5, 7};
Line(14) = {7, 2};
Line(15) = {2, 2};
Line(15) = {2, 3};
Line(16) = {3, 8};
Line(17) = {8, 6};
Line(18) = {6, 4};
Line(19) = {4, 1};
Curve Loop(1) = {12, 10, 18, 19};
Plane Surface(1) = {1};
Curve Loop(2) = {13, 11, 17, -10};
Plane Surface(2) = {2};
Curve Loop(3) = {14, 15, 16, -11};
Plane Surface(3) = {3};
Transfinite Curve {19, 10, 11, 15} = 3 Using Progression 1;
Transfinite Curve {13, 17} = 3 Using Progression 1;
Transfinite Curve {12, 18, 16, 14} = 2 Using Progression 1; # constant 2
Transfinite Surface {1} = {1, 4, 6, 5};
Transfinite Surface {2} = {5, 6, 8, 7};
Transfinite Surface {3} = {7, 8, 3, 2};
Recombine Surface {1};
Recombine Surface {2};
Recombine Surface {3};
glassObj[] = Extrude {0, 0, 4} {
Surface{1};
Layers{3};          // create only one layer of elements in the direction of extrusion
Recombine;};        // recombine triangular mesh to quadrangular mesh
fluidObj[] = Extrude {0, 0, 4} {
Surface{2};
Layers{3};          // create only one layer of elements in the direction of extrusion
Recombine;};        // recombine triangular mesh to quadrangular mesh
absObj[] = Extrude {0, 0, 4} {
Surface{3};
Layers{3};          // create only one layer of elements in the direction of extrusion
Recombine;};        // recombine triangular mesh to quadrangular mesh
// geometry boundaries must be named, partitions are found by neighboring cells
//+
Physical Surface("sflux1_hamb", 44) = {7};
//+
Physical Surface("noslip", 45) = {9, 11};
//+
Physical Surface("inlet", 46) = {13};
//+
Physical Surface("outlet", 47) = {2};
//+
Physical Surface("fflux1_noslip_s2s1", 48) = {10};
//+
Physical Surface("noslip_s2s2", 49) = {5};
//+
Physical Surface("none", 50) = {1, 3, 4, 6, 8, 14, 15, 16, 18};
//+
Physical Volume("solid_glass", 51) = {1};
Physical Volume("fluid_udara", 52) = {2};
Physical Volume("solid_absorber", 53) = {3};
Mesh.SaveAll = 1;