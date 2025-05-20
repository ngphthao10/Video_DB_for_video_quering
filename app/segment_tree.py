import numpy as np
from typing import Dict, Set, List, Optional, Union, Any

class FrameSegmentTree:

    def __init__(self, n: int, object_class: Optional[int] = None):
        self.n = n
        self.object_class = object_class
        self.height = int(np.ceil(np.log2(n))) + 1
        self.max_size = 2 * (2 ** self.height) - 1
        self.st = [set() for _ in range(self.max_size)]
        
    def _build_segment_tree(self, annotations: Dict[int, List[Dict]], node: int, start: int, end: int):

        if start == end:
            if start in annotations:
                for obj in annotations[start]:
                    if self.object_class is None or obj['class_id'] == self.object_class:
                        self.st[node].add(obj['_id'])
            return
            
        mid = (start + end) // 2
        self._build_segment_tree(annotations, 2*node+1, start, mid)
        self._build_segment_tree(annotations, 2*node+2, mid+1, end)
        
        self.st[node] = self.st[2*node+1].union(self.st[2*node+2])
    
    def build(self, annotations: Dict[int, List[Dict]]):
        self._build_segment_tree(annotations, 0, 0, self.n-1)
    
    def _query(self, node: int, start: int, end: int, l: int, r: int) -> Set:
        if start > r or end < l:
            return set()
            
        if l <= start and end <= r:
            return self.st[node]
            
        mid = (start + end) // 2
        left_query = self._query(2*node+1, start, mid, l, r)
        right_query = self._query(2*node+2, mid+1, end, l, r)
        
        return left_query.union(right_query)
    
    def query(self, l: int, r: int) -> Set:
        if l < 0 or r >= self.n or l > r:
            raise ValueError("Invalid query range")
        return self._query(0, 0, self.n-1, l, r)
    
    def to_dict(self) -> Dict[str, Any]:
        serialized_st = [list(s) for s in self.st]
        return {
            'n': self.n,
            'object_class': self.object_class,
            'height': self.height,
            'max_size': self.max_size,
            'st': serialized_st
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameSegmentTree':
        tree = cls(data['n'], data['object_class'])
        tree.height = data['height']
        tree.max_size = data['max_size']
        tree.st = [set(s) for s in data['st']]
        return tree