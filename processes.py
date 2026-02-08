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

        return map[node]

    def connected(self, srcNode, node) -> bool:
        if node in map[srcNode]:
            return True

        else: return False


class TimeStamp:
    def __init__(self, _init: int, _term: int, _dis=None):
        # _dis = {"type": TimeStamp}
        self.ts = {"initialized": _init, "terminated": _term}
        
        if _dis:
            self.ts["disruptions"] = []
            for i in _dis:
                self.ts["disruptions"][i] = _dis[i]


class Process:
    def __init__(self, name: "str", timeline: TimeStamp):
        self.name = name
        slef.timeline = timeline

