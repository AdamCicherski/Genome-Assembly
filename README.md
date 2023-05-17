# Genome-Assembly

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Quick start](#Quick-start)
* [Algorithm Explanation](#Algorithm-Explanation)

## General info
This project is simple genome de Novo assembler based on de Brujin Graphs aproach.
	
## Technologies
* Numpy >=1.21.1
* Numba >= 0.54.0
	
## Quick start
To run genome assembler type:

```
$ python assembly.py reads.fasta output.fasta
```
## Algorithm Explanation
In this section, we will provide a detailed explanation of how the algorithm works.

### Graph Construction and Read Correction
Before constructing the graph, the reads are corrected using a histogram-based approach. The corrected reads are then used to construct a de Bruijn graph with an initial value of k.

### Graph Compression
The first step in graph transformation is the replacement of unbranched paths with singular vertices. This operation, known as graph compression, helps simplify the graph structure. Following the compression, all tips (short paths with weak coverage branching from the main path of the graph) are removed. After this step, the graph is compressed once again.

### Bubble Structure Detection
The next step involves identifying bubble structures within the graph. Bubble structures are fragments of paths with the same starting and ending node, and no other common vertices between them. A BFS-based search algorithm is employed to find these bubble structures.

### Bubble Similarity Calculation and Pruning
Similarity between two branches of a bubble is calculated using the Needelman-Wunsch algorithm. If the similarity is high enough, the branch with weaker coverage is removed. After this pruning operation, all tips are once again removed, and the graph is compressed.

### Label Appending and Iteration
Labels of vertices with a length greater than the input read size are appended to the corrected reads list. A new iteration then begins, this time without histogram correction but with a higher value of k. The process of graph construction, compression, tip removal, bubble detection, similarity calculation, and pruning is repeated in each iteration.

### Scaffold Generation
After the last iteration, the labels of vertices are returned as scaffolds. If there are too many scaffolds, a greedy heuristic is used to find paths with higher coverage, which are then utilized as scaffolds.

