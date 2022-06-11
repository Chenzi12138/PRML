class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, name):
        self.children.append(Node(name))
        return self

    def bsf(self, result_array: list = []) -> list:
        queue = [self]
        while len(queue) > 0:
            current_node = queue.pop(0)
            result_array.append(current_node)
            queue.extend(current_node.children)
        return result_array


root = Node("A")
root.add_child("B")
root.add_child("C")
root.add_child("D")
root.children[0].add_child("E")
root.children[0].add_child("F")
root.children[2].add_child("G")
root.children[2].add_child("H")
root.children[0].children[1].add_child("I")
root.children[0].children[1].add_child("J")
root.children[2].children[0].add_child("K")

result = root.bsf()
print(result)