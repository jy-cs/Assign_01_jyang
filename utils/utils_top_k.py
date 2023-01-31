import sys
import nltk
from nltk.corpus import wordnet as wn
import numpy
import heapq
# Implementation of Customized Max Heap
class MaxHeap_MED_Word:
    
    def __init__(self, maxsize):
        
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [[]] * self.maxsize
        self.Root = 0
    
    # Function to return True if self is an empty heap; False, otherwise
    def is_empty(self):

        return self.size == 0

    # Function to return True if self is full; False, otherwise
    def is_full(self):

        return self.size == self.maxsize

    # Function to return the index of the last index
    def last(self):

        return self.size - 1
        
    # Function to return the position of parent for the node currently at pos
    def parent(self, pos):
        
        return (pos - 1) // 2

    # Function to return the position of the left child for the node currently at pos
    def leftChild(self, pos):
        
        return 2 * pos + 1

    # Function to return the position of the right child for the node currently at pos
    def rightChild(self, pos):
        
        return 2 * (pos + 1)
    
    # Function to return True if pos has left child; False, otherwise
    def has_leftChild(self, pos):

        return self.leftChild(pos) < self.size;

    # Function to return True if pos has right child; False, otherwise
    def has_rightChild(self, pos):
        
        return self.rightChild(pos) < self.size;

    # Function that returns true if the passed node is a leaf node
    def isLeaf(self, pos):
        
        return self.leftChild(pos) >= self.size and pos < self.size

    # Function that returns true if the passed node is a leaf node
    def isRoot(self, pos):
        
        return pos == 0

    # Function to swap two nodes of the heap
    def swap(self, pos_1, pos_2):
        
        tmp = self.Heap[pos_1]
        self.Heap[pos_1] = self.Heap[pos_2]
        self.Heap[pos_2] = tmp
        return

    # Function to heapify the node at pos
    def maxHeapify(self, pos):

        while not self.isLeaf(pos):
            large_idx = self.leftChild(pos)
            if self.has_rightChild(pos) and (self.Heap[self.rightChild(pos)][0] >
                                             self.Heap[large_idx][0]):
                large_idx = self.rightChild(pos)
            if self.Heap[pos][0] >= self.Heap[large_idx][0]:
                break
            self.swap(pos, large_idx)
            pos = large_idx
        return

    # Function to insert a node into the heap
    def insert(self, element):
        
        if self.size >= self.maxsize:
            return
        self.size += 1
        self.Heap[self.size - 1] = element

        current = self.size -1

        while (not self.isRoot(current) and 
               self.Heap[current][0] > self.Heap[self.parent(current)][0]):
            self.swap(current, self.parent(current))
            current = self.parent(current)
        return
    
    # Function to return the maximum element from the heap
    def top(self):

        return self.Heap[self.Root]
    
    # Function to return the maximum element's key from the heap
    def topKey(self):

        return self.Heap[self.Root][0]

    # Function to remove and return the maximum element from the heap
    def extractMax(self):

        popped = self.Heap[self.Root]
        self.Heap[self.Root] = self.Heap[self.last()]
        self.size -= 1
        if not self.is_empty():
            self.maxHeapify(self.Root)
        
        return popped

    # Function to print the contents of the heap
    def Print(self):
        
        for i in range(0, self.size):
            sl = self.Heap[i]
            if self.has_rightChild(i):
                lc = self.Heap[self.leftChild(i)]
                rc = self.Heap[self.rightChild(i)]
                print("PARENT-" + str(i) + ": (" + str(sl[0]) + "," + sl[1] +
                          ") ->LEFT CHILD: (" + str(lc[0]) + "," + lc[1] +
                          ") ->RIGHT CHILD: (" + str(rc[0]) + ", " + rc[1] + ")")
            elif self.has_leftChild(i):
                lc = self.Heap[self.leftChild(i)]
                print("PARENT-" + str(i) + ": (" + str(sl[0]) + "," + sl[1] +
                          ") ->LEFT CHILD: (" + str(lc[0]) + "," + lc[1] + ")")
            elif self.isLeaf(i):
                print("PARENT(Leaf)-" + str(i) + ": (" + str(sl[0]) + "," + sl[1] + ")")


def find_top_k_similar(inc_str, k):
    result = MaxHeap_MED_Word(k) # result is a k-element max heap
    for synset in wn.words():
        word = synset
        cur_tuple = (min_edit_distance(word, inc_str), word)
        if not result.is_full():
            result.insert(cur_tuple)
        elif cur_tuple[0] < result.topKey():
            result.extractMax()
            result.insert(cur_tuple)
    return result



