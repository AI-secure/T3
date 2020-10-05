# tree object from stanfordnlp/treeLSTM
class TreeNode(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.relation = list()

    def add_child(self, child, relation=None):
        child.parent = self
        self.num_children += 1
        self.children.append(child)
        self.relation.append(relation)

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