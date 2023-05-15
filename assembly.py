#!/usr/bin/python3
from DeBruijnGraph import *
import sys
import logging
import time


def read_fasta(path):
    reads_list = []
    with open(path) as file:
        for line in file:
            if line[0] != ">":
                reads_list.append(line.rstrip('\n'))
    return reads_list


def write_fasta(name, contigs):
    with open(name, 'w') as file:
        for i, item in enumerate(contigs):
            file.write(">contig_" + str(i) + '\n')
            file.write(item + '\n' if i < len(contigs) - 1 else item)


def neighbors1mm(kmer, alpha):
    """ Generate all neighbors at Hamming distance 1 from kmer """
    neighbors = []
    for j in range(len(kmer) - 1, -1, -1):
        oldc = kmer[j]
        for c in alpha:
            if c == oldc: continue
            neighbors.append(kmer[:j] + c + kmer[j + 1:])
    return neighbors


def kmerHist(reads, k):
    """ Return k-mer histogram and average # k-mer occurrences """
    kmerhist = {}
    for read in reads:
        for kmer in [read[i: i + k] for i in range(len(read) - (k - 1))]:
            kmerhist[kmer] = kmerhist.get(kmer, 0) + 1
    return kmerhist


def correct1mm(read, k, kmerhist, alpha, thresh):
    """ Return an error-corrected version of read.  k = k-mer length.
        kmerhist is kmer count map.  alpha is alphabet.  thresh is
        count threshold above which k-mer is considered correct. """
    # Iterate over k-mers in read
    for i in range(len(read) - (k - 1)):
        kmer = read[i: i + k]
        # If k-mer is infrequent...
        if kmerhist.get(kmer, 0) <= thresh:
            # Look for a frequent neighbor
            for newkmer in neighbors1mm(kmer, alpha):
                if kmerhist.get(newkmer, 0) > thresh:
                    # Found a frequent neighbor; replace old kmer
                    # with neighbor
                    read = read[:i] + newkmer + read[i + k:]
                    break
    # Return possibly-corrected read
    return read


complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def complementary(read):
    s = ''.join(complement[i] for i in read)
    return s[::-1]


def complementary_reads(reads):
    res = []
    for read in reads:
        res.append(complementary(read))
    return reads + res


def debrujin_iter(reads, start=8, stop=60, step=1, input_size=80):
    for k in range(start, stop, step):

        # Transform graph
        graph = DeBruijnGraph(reads, k)
        graph.simplyfy()
        graph.remove_tips()
        graph.simplyfy()
        graph.bubles_finder()
        graph.simplyfy()
        graph.remove_tips()
        graph.simplyfy()
        contigs = []

        #Add labels at least as long as input reads to the input of next iteration
        for i in graph.nodes.keys():
            if len(i) > input_size:
                contigs.append(i)
        new_reads = contigs
        for i in reads:
            for j in contigs:
                if i in j: break
            else:
                new_reads.append(i)
        reads = new_reads
    return graph


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Assembly started')
    try:
        input_file = sys.argv[1]
        output = sys.argv[2]
    except FileExistsError:
        logging.error('Wrong file path')
        sys.exit()
    k = 15
    reads = read_fasta(input_file)
    logging.info('{} reads fetched from  file {}'.format(len(reads), input_file))
    kmerhist = kmerHist(reads, k)
    corected = []
    for read in reads:
        corected.append(correct1mm(read, k, kmerhist, alpha='ACTG', thresh=2))
    logging.info('Reads corrected with histogram')
    logging.info('Starting main algorithm loop')
    graph = debrujin_iter(corected)
    if len(graph.nodes) > 5:  # Totally arbitrary choice
        graph.greedy()
    contigs = graph.nodes.keys()
    write_fasta(output, contigs)
    logging.info('Contigs written to file {}'.format(output))
    logging.info('Assembly done!')


if __name__ == "__main__":
    main()
