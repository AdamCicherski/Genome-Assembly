from collections import defaultdict
import heapq
import numpy as np
import time
import numba as nb


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('function {} performance {} sec '.format(method.__name__, (te - ts).__round__(2)))
        return result

    return timed


# @timeit
@nb.jit()
def needleman_wunsch(seq1, seq2):
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((m + 1, n + 1))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = 1 if seq1[j - 1] == seq2[i - 1] else 0
            match = score[i - 1][j - 1] + match_score
            delete = score[i - 1][j]
            insert = score[i][j - 1]
            score[i][j] = max(match, delete, insert)
    return score[i][j] / max(len(seq1), len(seq2))


class DeBruijnGraph(object):
    """ De Bruijn directed multigraph built from a collection of
        strings. User supplies strings and k-mer length k.  Nodes
        are k-1-mers.  An Edge corresponds to the k-mer that joins
        a left k-1-mer to a right k-1-mer. """

    @staticmethod
    def chop(st, k):
        """ Chop string into k-mers of given length """
        for i in range(len(st) - (k)):
            yield st[i:i + k + 1], st[i:i + k], st[i + 1:i + k + 1]

    class Node(object):

        """ Node representing a k-1 mer."""

        def __init__(self, km1mer):
            self.km1mer = km1mer
            self.coverage = 0

        def __lt__(self, other):
            return len(self.km1mer) < len(other.km1mer)

        def get_km1er(self):
            return self.km1mer

        def __hash__(self):
            return hash(self.km1mer)

        def __str__(self):
            return self.km1mer

    def __init__(self, str_iter, k, circularize=False):
        """ Build de Bruijn multigraph given string iterator and k-mer
            length k """

        self.successors = defaultdict(lambda: defaultdict(int))  # multimap from nodes to neighbors
        self.predecessors = defaultdict(lambda: defaultdict(int))
        self.nodes = {}  # maps k-1-mers to Node objects
        self.k = k
        for st in str_iter:
            if circularize:
                st += st[:k - 1]
            for kmer, km1L, km1R in self.chop(st, k):
                node_left, node_right = None, None
                if km1L in self.nodes:
                    node_left = self.nodes[km1L]
                else:
                    node_left = self.nodes[km1L] = self.Node(km1L)
                if km1R in self.nodes:
                    node_right = self.nodes[km1R]
                else:
                    node_right = self.nodes[km1R] = self.Node(km1R)
                self.successors[node_left][node_right] += 1
                self.predecessors[node_right][node_left] += 1

    def remove_node(self, node):
        # Maybe this will be usefull?
        if node in self.successors.keys():
            for i in self.successors[node]:
                del self.predecessors[i][node]
                if self.predecessors[i] == {}:
                    del self.predecessors[i]
            del self.successors[node]
        if node in self.predecessors.keys():
            for i in self.predecessors[node]:
                del self.successors[i][node]
                if self.successors[i] == {}:
                    del self.successors[i]
            del self.predecessors[node]
        del self.nodes[node.get_km1er()]

    def remove_outgoing_tip(self, node, weight):

        """funkcja do usuwania tipow nie majacych zadnego wyjscia"""
        if len(node.get_km1er()) > 2 * self.k or sum(self.predecessors[node].values()) >= weight:
            return False
        else:
            przodkowie = self.predecessors[node]
            self.remove_node(node)
            for i in przodkowie:
                if i not in self.successors.keys():
                    self.remove_outgoing_tip(i, weight)

    def remove_ingoing_tip(self, node, weight):
        """funkcja do usuwania tipow nie majacych zadnego wejscia"""
        if len(node.get_km1er()) > 2 * self.k or sum(self.successors[node].values()) >= weight:
            return False
        else:
            synowie = self.successors[node]
            self.remove_node(node)
            for i in synowie:
                if i not in self.predecessors.keys():
                    self.remove_ingoing_tip(i, weight)

    def remove_tips(self, weight=3):
        """funkcja do usuwania tipow"""
        for i in list(filter(lambda x: x not in self.predecessors.keys(), self.nodes.values())):
            self.remove_ingoing_tip(i, weight)
        for i in list(filter(lambda x: x not in self.successors.keys(), self.nodes.values())):
            self.remove_outgoing_tip(i, weight)

    def marge_nodes(self, father, son):

        # Merrge nodes labels
        new_label = father.get_km1er() + son.get_km1er()[self.k - 1:]  # again here was k-2
        new_node = self.Node(new_label)
        self.nodes[new_label] = new_node
        new_node.coverage = father.coverage + son.coverage + self.successors[father][son]
        return new_node

    def move_edges(self, father, son, new_node):
        # Move old edges to new node
        if son in self.successors.keys():
            if father in self.successors[son].keys():
                self.successors[new_node][new_node] = self.successors[son][father]
                del self.successors[son][father]
            self.successors[new_node] = self.successors[son]
            for x in self.successors[son].keys():
                self.predecessors[x][new_node] = self.predecessors[x][son]
        if father in self.predecessors.keys():
            if son in self.predecessors[father].keys():
                self.predecessors[new_node][new_node] = self.predecessors[father][son]
                del self.predecessors[father][son]
            self.predecessors[new_node] = self.predecessors[father]
            for x in self.predecessors[father].keys():
                self.successors[x][new_node] = self.successors[x][father]

    # @timeit
    def simplyfy(self):
        l = list(self.nodes.values())
        visited = {}
        queue = []
        for node in l:
            if node not in visited:
                queue.append(node)
            while queue:
                father = queue.pop()
                visited[father] = True
                if father not in self.successors.keys():
                    continue
                if len(self.successors[father]) != 1:
                    neighbor = list(filter(lambda x: x not in visited, self.successors[father].keys()))
                    queue += neighbor
                    for i in neighbor:
                        visited[i] = True
                    continue
                son = list(self.successors[father].keys())[0]
                if father == son:  # if father node has only loop to itself
                    continue
                if len(self.predecessors[son]) != 1:
                    if son not in visited:
                        queue.append(son)
                        visited[son] = True
                    continue

                new_node = self.marge_nodes(father, son)
                self.move_edges(father, son, new_node)
                self.remove_node(father)
                self.remove_node(son)

                # dodajemy nowy nod do kolejki
                queue.insert(0, new_node)
                visited[new_node] = True

    def traceback_buble(self, x, end, father):
        y = father[end]
        x_path = [x]
        y_path = [y]
        while True:

            if x == end:
                return False
            if x in y_path:
                lca = x
                break
            elif y in x_path:
                lca = y
                break
            else:
                x = father[x]
                y = father[y]
                x_path.append(x)
                y_path.append(y)
            if x is None and y is None:
                return False
        x_path.insert(0, end)
        y_path.insert(0, end)
        return x_path[:x_path.index(lca) + 1], y_path[:y_path.index(lca) + 1]

    def read_paths_seq(self, path):  # read sequence from the path
        return path[0].get_km1er() + ''.join(node.get_km1er()[self.k - 1:] for node in path[1:])

    def del_path2(self, path, father):
        for i in path:
            self.remove_node(i)
            del father[i]

        # @timeit

    def search_for_bubble(self, node, visited):
        father = defaultdict(lambda: None)
        dist = {}
        for x in self.nodes.values():  # Set distance to each node to infinity
            dist[x] = np.inf
        dist[node] = 0  # Set distance to starting node to 0
        queue = [(0, node)]
        visited.add(node)
        while queue:
            node = heapq.heappop(queue)[1]
            for i, weight in self.successors[node].items():
                visited.add(i)
                if dist[i] == np.inf:  # Was not visited in previous steps
                    dist[i] = dist[node] + (len(i.get_km1er()) - self.k + 2) / (weight + i.coverage)
                    heapq.heappush(queue, (dist[i], i))
                    father[i] = node
                elif dist[i] == 0:  # Loop to the starting node
                    continue
                else:  # Was visited on different path
                    paths = self.traceback_buble(node, i, father)
                    if paths:
                        seq1 = self.read_paths_seq(paths[0][::-1])
                        seq2 = self.read_paths_seq(paths[1][::-1])
                        score = needleman_wunsch(seq1, seq2)
                        if score < 0.8:
                            continue  # Sequences to different for merge
                        elif dist[i] > dist[node] + (len(i.get_km1er()) - self.k + 2) / (weight + i.coverage):
                            dist[i] = dist[node] + (len(i.get_km1er()) - self.k + 2) / (weight + i.coverage)
                            heapq.heappush(queue, (dist[i], i))
                            self.del_path2(paths[1][1:-1], father)
                            father[i] = node
                        else:
                            self.del_path2(paths[0][1:-1], father)
                            break

    def bubles_finder(self):
        visited = set()
        for node in list(self.nodes.values()):
            if node not in visited:  # jesli nod nie byl odwiedzony w poprednich przeszukiwaniach
                self.search_for_bubble(node, visited)

    def greedy(self):
        while True:
            best = 0
            for i in self.successors:  # szukamy krawedzi nie bedacej loopem o najwiekszej wadze
                for j in self.successors[i]:
                    if self.successors[i][j] > best and i != j:
                        best = len(i.get_km1er() + j.get_km1er()[self.k - 1:])  # self.successors[i][j]
                        trace_i = i
                        trace_j = j
            if best == 0: break  # nie ma juz zadnych dobryh krwedzi

            # tworzymy nowy nod i usuwamy stare podobnie jak w simplyfy
            new_node = self.marge_nodes(trace_i, trace_j)
            self.move_edges(trace_i, trace_j, new_node)

            self.remove_node(trace_i)
            self.remove_node(trace_j)

            for i in list(self.successors.keys()):
                if self.successors[i] == {}:
                    del self.successors[i]
