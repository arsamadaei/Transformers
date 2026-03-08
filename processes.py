import random
import json
import weakref
import os

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

    @property
    def _init(self):
        return self.ts["initialized"]
    
    @_init.setter
    def _init(self, value):
        self.ts["initialized"] = value
    
    @property
    def _term(self):
        return self.ts["terminated"]
    
    @_term.setter
    def _term(self, value):
        self.ts["terminated"] = value

    
    def add_disruption(self, _type: str, timeStamp: "TimeStamp"):
        self.ts["disruptions"].append({_type: timeStamp})


    def serialize_disruptions(self) -> list:
        """
        Converts disruptions into a list serializable by json, since _dis is a dictionary
        containing a TimeStamp object, which is not parsable for json.
        """
        serializedList = []

        for i in self.ts["disruptions"]:
            for k, v in i.items():
                serializedList.append({k: {"initialized": v.ts.get("initialized", 0),
                                     "terminated": v.ts.get("terminated", -1)}})
        
        return serializedList


    def to_hirearchial_dict(self):
        return { "initialized": self.ts["initialized"],
                 "terminated": self.ts["terminated"],
                 "disruptions": self.serialize_disruptions()
        }
    
    @staticmethod
    def de_serialize(dis: list) -> dict:
        _dis = {}
        for i in dis:
            for t, ts in i.items():
                _dis[t] = TimeStamp(
                        ts.get("initialized", 0),
                        ts.get("terminated", -1)
                        )

        return _dis

def create_id(epoch: int, layer: int, len: int=8) -> str:
    hex_chars = '0123456789abcdef'
    randID = ''.join(random.choices(hex_chars, k=len))

    return f"{epoch}x{layer}-{randID}"


class Process(TimeStamp):
    __instances = weakref.WeakSet()

    def __init__(self, name: str,
                 _init: float,
                 _term: float = -1,
                 layer: int = 0,
                 epoch: int = 0,
                 _dis=None,
                 parent_uid: str|None = None,
                 subtasks: list|None = None):
        # layer = 0 means the process does not inhibit parallel processing
        # _term = -1 means the process is ongoing
        # epoch tracks which training epoch this process belongs to
        super().__init__(_init, _term, _dis)
        self.name = name
        self.layer = layer
        self.epoch = epoch

        # Generate process ID with epoch and layer
        self.uid = create_id(self.epoch, self.layer)
        self.parent_uid = parent_uid
        self.subtasks = subtasks if subtasks is not None else []

        Process.__instances.add(self)

    def add_subtask(self, process):
        process.parent_uid = self.uid
        self.subtasks.append(process.uid)


    def to_hirearchial_dict(self) -> dict:
        """
        Converts process to json-style hirearchial dictionary suited for json formatting.
        """
        return { "name"    : self.name,
                 "uid"     : self.uid,
                 "parent_uid": self.parent_uid,
                 "epoch"   : self.epoch,
                 "layer"   : self.layer,
                 "timeline": super().to_hirearchial_dict(),
                 "subtasks": self.subtasks
                }

    def to_storage_dict(self):
        """
        Returns tuple of (uid, process_dict) for storage in dictionary format.
        Enables O(1) lookup by uid.
        """
        return (self.uid, self.to_hirearchial_dict())

    def write(self, ofile: str="eval_results/processes.json"):
        """
        Write process information to json file.
        Uses dictionary structure for O(1) lookup by uid.
        
        ofile: output file location
        """

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(ofile), exist_ok=True)

        if os.path.exists(ofile) and os.path.getsize(ofile) > 0:
            try:
                with open(ofile, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {"processes": {}}
        else:
            data = {"processes": {}}

        processes = self.to_hirearchial_dict()

        # O(1) check if process exists - direct dict lookup
        if self.uid in data["processes"]:
            if data["processes"][self.uid].get("timeline") == self.to_hirearchial_dict()["timeline"]:
                return ofile

            else: 
                data["processes"][self.uid] = processes
    
                with open(ofile, 'w+') as f:
                    json.dump(data, f, indent=2)

                return ofile


        data["processes"][self.uid] = processes
        with open(ofile, 'w') as f:
            json.dump(data, f, indent=2)
        

        return ofile


    @staticmethod
    def get_epoch_from_uid(uid):
        """Extract epoch from UID format: epochxlayer-id (e.g., '5x2-abc123')."""
        if "x" in uid and "-" in uid:
            try:
                return int(uid.split("x")[0])
            except ValueError:
                pass
        return 0
    
    @staticmethod
    def get_layer_from_uid(uid):
        """Extract layer from UID format: epochxlayer-id (e.g., '5x2-abc123')."""
        if "x" in uid and "-" in uid:
            try:
                layer_part = uid.split("x")[1]
                return int(layer_part.split("-")[0])
            except (ValueError, IndexError):
                pass
        return 0

    @classmethod
    def load_dict(cls, data: dict) -> "Process":
        """
        Read json format dictionary and create a series of process objects.
        """
        name = data.get("name")
        uid = data.get("uid")
        timeStamp = data.get("timeline")
        _init = timeStamp.get("initialized")
        _term = timeStamp.get("terminated")
        dis: list = timeStamp.get("disruptions", [])
        _dis = TimeStamp.de_serialize(dis)
        puid = data.get("parent_uid")
        sub_tasks = data.get("subtasks")
        epoch = data.get("epoch", cls.get_epoch_from_uid(uid))
        layer = data.get("layer", cls.get_layer_from_uid(uid))
        
        process = cls( name, 
                       _init,
                       _term,
                       layer,
                       epoch,
                       _dis,
                       puid,
                       sub_tasks
                      )

        return process
    
    @classmethod
    def storeAll(cls, ofile: str="eval_results/processes.json"):
        """
        Store processes by merging with existing file.
        Only adds/updates changed processes, preserves existing data.
        """

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(ofile), exist_ok=True)

        if os.path.exists(ofile) and os.path.getsize(ofile) > 0:
            try:
                with open(ofile, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {"processes": {}}
        else:
            data = {"processes": {}}
        
        for p in cls.__instances:
            data["processes"][p.uid] = p.to_hirearchial_dict()
        
        with open(ofile, 'w+') as f:
            json.dump(data, f, indent=2)

        return ofile

    @classmethod
    def loadAll(cls, ofile: str="eval_results/processes.json"):
        """
        Load all processes in the json file into a list of processes.
        """

        if not os.path.exists(ofile) or os.path.getsize(ofile) == 0:
            return []
        
        # Create directory if it doesn't exist (for future writes)
        os.makedirs(os.path.dirname(ofile), exist_ok=True)

        try:
            with open(ofile, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return []

        return [cls.load_dict(p) for p in data.get("processes", {}).values()]

    @classmethod
    def load(cls, uid: str, ofile: str="eval_results/processes.json"):
        """
        Load a specific process by UID.
        """
        if not os.path.exists(ofile) or os.path.getsize(ofile) == 0:
            return None
        
        # Create directory if it doesn't exist (for future writes)
        os.makedirs(os.path.dirname(ofile), exist_ok=True)

        try:
            with open(ofile, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return None

        # O(1) lookup - direct dictionary access
        process_data = data.get("processes", {}).get(uid, {})
        
        return cls.load_dict(process_data) if process_data else None

    @staticmethod
    def remove(uid: str, ofile: str="eval_results/processes.json"):
        """
        Remove a specific process by UID.
        """

        if not os.path.exists(ofile):
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(ofile), exist_ok=True)
        
        with open(ofile, 'r') as f:
            data = json.load(f)

        if uid in data.get("processes", {}):
            del data["processes"][uid]

            with open(ofile, 'w+') as f:
                json.dump(data, f, indent=2)
            
            return True
        
        return False

    @classmethod
    def get_topLevelTasks(cls, ofile = None):
        if ofile is not None:
            all_processes = cls.loadAll(ofile)
        
        else:
             all_processes = cls.loadAll()

        return [p for p in all_processes if p.parent_uid is None]

    @staticmethod
    def clear_storage(ofile: str="eval_results/processes.json"):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(ofile), exist_ok=True)
        
        with open(ofile, 'w') as f:
            json.dump({"processes": {}}, f, indent=2)

    @staticmethod
    def load_parent(ofile = None):
        if self.parent_uid:
            return Process.load(self.parent_uid)

    @staticmethod
    def load_subtasks(ofile = None):
        return [Process.load(uid) for uid in self.subtasks]


