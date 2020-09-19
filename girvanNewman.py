import json
import os
import time
import sys
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import StorageLevel
import itertools
from collections import defaultdict
#os.environ['PYSPARK_PYTHON'] = 'C:\ProgramData\Anaconda3\python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:\ProgramData\Anaconda3'

import numpy as np
from itertools import combinations
"""
spark-submit task1.py ./data/yelp_train.csv ./outputTask1.txt
"""

from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *
from pyspark.sql import SQLContext
from pyspark.sql import functions as F

# from IPython.display import display


start=time.time()

sc = SparkContext('local[*]', 'task1')
sqlContext = SQLContext(sc)

#input_file_path = "./yelp_dataset/review.json"
sc.setLogLevel("ERROR")


input_file=sys.argv[1]
output_file1=sys.argv[2]
output_file2=sys.argv[3]

network=sc.textFile(input_file).map(lambda x:tuple(x.split(' ')))#.map(lambda x: (x[0],x[1]))
vertices=network.flatMap(lambda x:x).distinct().map(lambda x: x).collect()
edges = network.map(lambda x: (x[0], x[1])).collect()

print(vertices)

# vertices = sqlContext.createDataFrame(
#   vertices, ["id"])
#e=network.collect()
# edges = network.flatMap(lambda x: [(x[1], [x[0]])]).reduceByKey(lambda x, y: x + y)
# edges=edges.collect()
# edge_node_map={}

# for i in edges:
# 	edge_node_map[i[0]]=i[1]
vertice_node_map=defaultdict(set)
print(edges)
for i in edges:
	vertice_node_map[i[0]].add(i[1])
	vertice_node_map[i[1]].add(i[0])
vertice_node_map=dict(vertice_node_map)

print(vertice_node_map)


def bfs(root,vertice_node_map=vertice_node_map):
    queue = [root]
    
    depth={v:-1 for v in vertice_node_map.keys()}
    parents={v:[] for v in vertice_node_map.keys()}
    traversals={v:0 for v in vertice_node_map.keys()}
    depth[root] = 0
    traversals[root] = 1
    
    visited = [root]
    while queue:
        curr = queue[0]
        queue=queue[1:]
        
        children = vertice_node_map[curr]
        for child in children:
            if depth[child] == -1:
                queue.append(child)
                depth[child] = depth[curr] + 1
            if depth[child] == depth[curr] + 1:
                parents[child].append(curr)
                traversals[child] = traversals[child] + traversals[curr]
        visited.append(curr)
        #print(visited, parents, traversals)
    return visited, parents, traversals

def find_betweennees(vertices,vertice_node_map=vertice_node_map):
    betweenness = {}
    for v in vertices:
        visited, parents, traversals = bfs(v,vertice_node_map)

        edge_credit = {}
        vertex_credit={}
        for j in visited:
            vertex_credit[j]=1
        
        for vertex in visited[::-1]:
            for p in parents[vertex]:
                set_ = min(vertex, p), max(vertex, p)
                total_add = vertex_credit[vertex] * traversals[p] / traversals[vertex]
                if set_ not in edge_credit:
                    edge_credit[set_] = 0
                vertex_credit[p] += total_add
                edge_credit[set_] += total_add  

        for edge, c in edge_credit.items():
            if edge not in betweenness:
                betweenness[edge] = 0
            betweenness[edge] += c / 2
    return sorted(betweenness.items(), key=lambda x: (-x[1]))

#print(vertices)
#print(vertice_node_map)

def modularity(communities, graph, edges_count):
    mod = 0
    for com in communities:
        for i in com:
            for j in com:
                if j in graph[i]:
                    jval=1
                else: jval=0
                mod+=jval-float(len(graph[i])*len(graph[j]))/float(2*edges_count)
                
    return mod / (2 * edges_count)

def find_community(betweenness,vertices):
    betweenness_=betweenness.copy()
    vertice_node_map_=vertice_node_map.copy()
    vertices=set(vertices)
    mcommunities=[]
    edges_count=len(betweenness)
    mmod=-1

    while betweenness_:
        print(betweenness_)
        communities = []
        vertices_temp=vertices.copy()
        print(vertices_temp)
        while vertices_temp:
            temp_com=set()
            queue = [vertices_temp.pop()]
            
            while queue:
                #print(queue)
                curr = queue[0] 
                queue=queue[1:]
                temp_com.add(curr)
                queue.extend([temp_node for temp_node in vertice_node_map_[curr] if temp_node not in temp_com])
                            
            communities.append(sorted(list(temp_com)))
            #print(type(vertices_temp),type(community))
            vertices_temp = vertices_temp.difference(temp_com)
        print(communities)

        mod=modularity(communities,vertice_node_map,edges_count)
        print('modularities:',mmod,mod)
        if mmod<mod:
            mcommunities=communities
            mmod=mod
        print(mod)
        mbetweeness=max([r for i,r in betweenness_])
        removed_edges = [edge for edge, value in betweenness_ if value == mbetweeness]
        for edge in removed_edges:
            vertice_node_map_[edge[0]]={i for i in vertice_node_map_[edge[0]] if i!=edge[1]}
            vertice_node_map_[edge[1]]={i for i in vertice_node_map_[edge[1]] if i!=edge[0]}

        betweenness_=find_betweennees(vertices,vertice_node_map_)
    return mcommunities    
final_betweenness=find_betweennees(vertices)


communities=find_community(final_betweenness,vertices)


print(len(communities))


f=open(output_file1,'w')
for i in final_betweenness:
	f.write(str(i[0])+', '+str(i[1])+'\n')

f.close()
communities=sorted(communities,key=lambda x:(len(x),x[0]))
f=open(output_file2,'w')
for i in communities:
#    print(i)
    f.write(str(i)[1:-1] + "\n")
f.close()
