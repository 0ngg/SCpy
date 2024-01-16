SetFactory("OpenCASCADE");

// 2282 nodes, 3940 elements

// input
base = 1.5; // width
height = 0.5;
length = 2.5;
m_abs = 0.022;
m_glass = 0.022;
m_insul = 0.050;
M = 8; N = 8; O = 2;

Point(1) = {-base/2, 0, -height/2};
Point(2) = {base/2, 0, -height/2};
Point(3) = {-base/2, 0, height/2};
Point(4) = {base/2, 0, height/2};

Point(5) = {-base/2, 0, -height/2 - m_abs};
Point(6) = {base/2, 0, -height/2 - m_abs};
Point(7) = {-base/2, 0, height/2 + m_glass};
Point(8) = {base/2, 0, height/2 + m_glass};

Line(1) = {8, 4};
Line(2) = {4, 2};
Line(3) = {2, 6};
Line(4) = {6, 5};
Line(5) = {5, 1};
Line(6) = {1, 3};
Line(7) = {3, 7};
Line(8) = {7, 8};
Line(9) = {4, 3};
Line(10) = {1, 2};

Curve Loop(1) = {8, 1, 9, 7};
Plane Surface(1) = {1};
Curve Loop(2) = {2, -10, 6, -9};
Plane Surface(2) = {2};
Curve Loop(3) = {3, 4, 5, 10};
Plane Surface(3) = {3};

Transfinite Curve {8, 9, 10, 4} = M Using Progression 1;
Transfinite Curve {1, 7, 5, 3} = O Using Progression 1;
Transfinite Curve {2, 6} = N Using Progression 1;
Transfinite Surface {1} = {8, 4, 3, 7} Alternated;
Transfinite Surface {2} = {4, 2, 1, 3} Alternated;
Transfinite Surface {3} = {2, 6, 5, 1} Alternated;

Physical Surface("glass", 11) = {1};
Physical Surface("fluid", 12) = {2};
Physical Surface("abs", 13) = {3};
Physical Curve("hamb", 14) = {8};
Physical Curve("qglass", 15) = {8};
Physical Curve("qamb", 16) = {10};
Physical Curve("htop", 17) = {10};
Physical Curve("hbot", 18) = {9};
Physical Curve("noslip", 19) = {9, 2, 10, 6};

Point(9) = {-base/2, 0, -height/2 - m_abs - m_insul};
Point(10) = {base/2, 0, -height/2 - m_abs - m_insul};

Line(11) = {6, 10};
Line(12) = {10, 9};
Line(13) = {9, 5};

Curve Loop(4) = {11, 12, 13, -4};
Plane Surface(4) = {4};

Transfinite Curve {11, 13} = O Using Progression 1;
Transfinite Curve {12} = M Using Progression 1;
Physical Surface("insul", 20) = {4};
Physical Curve("room", 21) = {12};

Mesh.Algorithm = 6;
	Recombine Surface{:};
Mesh.SaveAll = 1;



