#command to run the code
#python first.py power_input.txt first_output.txt
#python first1.py data_sets\myexample.csv firstopt3.csv

# export PYSPARK_DRIVER_PYTHON=jupyter
# export PYSPARK_DRIVER_PYTHON_OPTS=notebook
# pyspark --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11

#/home/local/spark/latest/bin/spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 first.py ../resource/asnlib/publicdata/power_input.txt output.txt
from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
import pyspark
import time
import sys

#os.environ["PYSPARK_SUBMIT_ARGS"] = ( "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11") 

#timer start
start_time=time.time()

#creating a spark context
conf = pyspark.SparkConf().setMaster("local[*]").setAppName("first").setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '3'), ('spark.cores.max', '3'), ('spark.driver.memory','8g')])
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
#creating a spark context
#sc = SparkContext('local[*]','first')
sc.setLogLevel('ERROR')

#take command line inputs
input_path = sys.argv[1]
output_path = sys.argv[2]

#reading the dataset into an RDD
originalRDD=sc.textFile(input_path).map(lambda line: line.split(" ")).map(lambda line:(line[0],line[1])).persist(pyspark.StorageLevel.DISK_ONLY)

#for x in originalRDD.collect():
#   print(x)

nodesRDD=originalRDD.flatMap(lambda line: [(line[0],),(line[1],)]).distinct()
Nodes=nodesRDD.collect()
Nodes=sqlContext.createDataFrame(Nodes,['id'])
#print(Nodes.head())

edgesRDD=originalRDD.flatMap(lambda line:[(line[0],line[1]),(line[1],line[0])]).distinct()
Edges=edgesRDD.collect()
Edges=sqlContext.createDataFrame(Edges,['src','dst'])
#print(Edges.head())

g=GraphFrame(Nodes,Edges)
print(g)

maxIterations=5

#g.vertices.show()
#g.edges.show()

result = g.labelPropagation(maxIter=maxIterations)
result.show()

ResultsRDD=result.rdd.map(list).map(lambda line:(line[1],[line[0]])).reduceByKey(lambda a,b:a+b).map(lambda line:(line[1]))
results=ResultsRDD.collect()
#for x in ResultsRDD.take(10):
#    print(x)

#clusters=ResultsRDD.count()
#Results=ResultsRDD.takeOrdered(clusters,key=lambda x: len(x))

output=[]

for item in results:
    item.sort()
    item=tuple(item)
    output.append(item)

output.sort()
final_output=dict()
for x in output:
    if len(x) in final_output:
        final_output[len(x)].append(x)
    else:
        final_output[len(x)]=[x]
print(final_output)
    
print("Duration: %s" % (time.time() - start_time))