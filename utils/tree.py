# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def remove_child(self, child):
        child.parent = None
        self.num_children -= 1
        self.children.remove(child)
        self.children += child.children

    def all_children(self, children = []):
        children += self.children
        for c in self.children:
            c.all_children(children)
        return children

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def visual(self):
        print('%s\t%s' %(self.idx, [c.idx for c in self.children]))
        for c in self.children:
            if c.num_children > 0:
                c.visual()
