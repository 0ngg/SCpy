from typing import TextIO
from .mesh import Mesh
from .mesh_format_parser import MeshFormatParser
from .physical_parser import NameParser, EntityParser
from .nodes_parser import NodesParser
from .elements_parser import ElementsParser
from .abstract_parser import AbstractParser

DEFAULT_PARSERS = [
    MeshFormatParser,
    NameParser,
    EntityParser,
    NodesParser,
    ElementsParser
]

class MainParser(AbstractParser):

    """ The main parser class, using other parsers. """

    def __init__(self, parsers=DEFAULT_PARSERS):
        self.parsers = parsers

    def parse(self, mesh: Mesh, io: TextIO) -> None:
        for line in io:
            line = line.strip()
            for parser in self.parsers:
                if parser.get_section_name() == line:
                    try:
                        parser.parse(mesh, io)
                    except Exception:
                        print("Unable to parse section %s from mesh!" % line)
                        raise
        for i in mesh.physical_groups_.values():
            for j in i.get_blocks():
                mesh.element_entities_[j].set_group(i.get_name())
        for i in mesh.element_entities_.values():
            for j in i.get_elements():
                j.set_group(i.get_group())