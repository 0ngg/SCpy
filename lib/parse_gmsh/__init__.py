from .mesh import Mesh
#from .info import Info, Geom
from .main_parser import MainParser

__version__ = "0.2.0"
__author__ = "Jukka Aho <ahojukka5@gmail.com>" # thanks!


def parse(filename: str):
    """Parse Gmsh .msh file and return info object."""
    mesh = Mesh()
    mesh.set_name(filename)
    parser = MainParser()
    with open(filename, "r") as io:
        parser.parse(mesh, io)
    return mesh
