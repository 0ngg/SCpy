class PhysicalEntity(object):

    def __init__(self):
        self.dimension_ = -1
        self.tag_ = -1
        self.name_ = None
        self.blocks_ = []
    
    def set_tag(self, tag: int):
        self.tag_ = tag

    def get_tag(self):
        return self.tag_

    def set_dimension(self, dimension: int):
        self.dimension_ = dimension

    def get_dimension(self) -> int:
        return self.dimension_

    def set_name(self, name: str):
        self.name_ = name

    def get_name(self) -> str:
        return self.name_

    def set_blocks(self, block: tuple):
        self.blocks_.append(block)

    def get_blocks(self):
        return list((self.dimension_, i[0], i[1]) for i in self.blocks_)