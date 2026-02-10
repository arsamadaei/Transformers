import random

class Mindmap:
    def __init__(self, map: dict):
        """
        A mindmap can map the schematics of a process. Here, the mindmap can be captured through its map dictionary.
        The map dictionary includes a list of nodes, which are process names, and edges, which connect nodes together.
        """
        self.map = map

    def nodes(self):
        return self.map.keys()

    def edges(self):
        return self.map.values()

    def connections(self, node):
        """
        Returns nodes connected to the edges of a source node.
        """
        if node in self.map:
            return self.map[node]
        else:
            return []

    def connected(self, srcNode, node) -> bool:
        if srcNode in self.map and node in self.map[srcNode]:
            return True
        return False

    def connect(self,srcNode, node, direction: bool | None =None):
        """
        If direction is False: srcNode to node
        If direction is True: node to srcNode
        If direction is None: bidirectional connection
        """
        if direction is None or direction is True:
            if node not in self.map:
                self.map[node] = []

            self.map[node].append(srcNode)

        if direction is None or direction is False:
            if srcNode not in self.map:
                self.map[srcNode] = []

            self.map[srcNode].append(node)


class TimeStamp:
    def __init__(self, _init: int, _term: int, _dis=None):
        """
        _init: time of initiation.
        _term: time of termination.
        _dis: disruptions such as pausing a task, or async pausing.
        _dis = {"type": TimeStamp}
        """
        self.ts = {"initialized": _init, "terminated": _term}
        self.ts["disruptions"] = []
    
        if _dis is not None:
            for i in _dis:
                self.ts["disruptions"].append({i: _dis[i]})

    def add_disruption(self, _type: str, timeStamp: "TimeStamp"):
        self.ts["disruptions"].append({_type: timeStamp})


class Process(TimeStamp):
    def __init__(self, name: str, layer: int, _init: int, _term: int, _dis=None):
        super().__init__(_init, _term, _dis)
        self.name = name

        # Generate process ID
        hex_chars = '0123456789abcdef'
        randID = ''.join(random.choices(hex_chars, k=4))

        self.uid = f"{layer}-{randID}"

        
        


