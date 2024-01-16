from typing import TextIO
from .helpers import parse_physical
from .abstract_parser import AbstractParser
from .physical_entity import PhysicalEntity
from .mesh import Mesh

class NameParser(AbstractParser):

    @staticmethod
    def get_section_name():
        return "$PhysicalNames"

    @staticmethod
    def parse(mesh: Mesh, io: TextIO):
        line = io.readline()
        if line.startswith("$PhysicalNames"):
            line = io.readline()
        meta = list(map(int, line.strip().split(" ")))
        number_of_names = meta[0]
        for i in range(0, number_of_names):
            emeta = parse_physical(io)
            physical = PhysicalEntity()
            physical.set_dimension(emeta[0])
            physical.set_tag(emeta[1])
            physical.set_name(emeta[2])
            mesh.add_physical_group(physical)

class EntityParser(AbstractParser):

    @staticmethod
    def get_section_name():
        return "$Entities"

    @staticmethod
    def parse(mesh: Mesh, io: TextIO):
        face_type_dict = {3: 2, 4: 3}
        cell_type_dict = {6: 5}
        line = io.readline()
        if line.startswith("$Entities"):
            line = io.readline()
        meta = list(map(int, line.strip().split(" ")))
        number_of_point_entities = meta[0]
        number_of_curve_entities = meta[1]
        number_of_surface_entities = meta[2]
        if len(meta) == 4:
            number_of_cell_entities = meta[3]
        for i in range(0, number_of_point_entities):
            line = io.readline()
        for i in range(0, number_of_curve_entities):
            emeta = io.readline().strip().split(" ")
            if int(emeta[7]) > 0:
                block_id = int(emeta[0])
                physical_id = list(map(lambda j: int(j), emeta[8:(8+int(emeta[7]))]))
                for j in physical_id:
                    mesh.physical_groups_[j].set_blocks((block_id, 1))
        for i in range(0, number_of_surface_entities):
            emeta = io.readline().strip().split(" ")
            if int(emeta[7]) > 0:
                block_id = int(emeta[0])
                physical_id = list(map(lambda j: int(j), emeta[8:(8+int(emeta[7]))]))
                for j in physical_id:
                    type = int(emeta[7 + int(emeta[7]) + 1])
                    mesh.physical_groups_[j].set_blocks((block_id, face_type_dict[type]))
        for i in range(0, number_of_cell_entities):
            emeta = io.readline().strip().split(" ")
            if int(emeta[7]) > 0:
                block_id = int(emeta[0])
                physical_id = list(map(lambda j: int(j), emeta[8:(8+int(emeta[7]))]))
                for j in physical_id:
                    type = int(emeta[7 + int(emeta[7]) + 1])
                    mesh.physical_groups_[j].set_blocks((block_id, cell_type_dict[type]))