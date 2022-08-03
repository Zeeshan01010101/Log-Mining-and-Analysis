from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local[10]") \
        .appName("Movies Rating Data") \
        .config("spark.local.dir","/fastdata/acp21zgs") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

sc = spark.sparkContext

logFile = spark.read.text("NASA_access_log_Jul95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently
logFile.show(20, False)
# Here I am importing
import pyspark.sql.functions as F
from pyspark.sql.functions import date_format
from pyspark.sql.functions import col
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Here I am using regular Expression to filter out data
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
data.show(20,False)
# Here I am using Timestamp standard defined function to convert date into standard form
dataFrameWithTimeStamps = data.withColumn('date', F.date_format(F.to_timestamp('timestamp', 'dd/MMM/yyyy:HH:mm:ss Z'), 'yyyy-MM-dd HH:mm:ss Z'))
dataFrameWithTimeStamps.show()

# Here I am converting dates into week days
new_data_time_stamp = dataFrameWithTimeStamps.withColumn("week_day_full", date_format(col("date"), "EEEE"))
df = new_data_time_stamp.select(['HTTP reply code',
 'bytes in the reply',
 'date',
 'week_day_full'])
 
df.show()
# Here I am extracting dates day of week from date
daysss = df.withColumn("dayofweek",F.dayofweek(df['date']))
daysss.show()
 
from pyspark.sql.functions import lit
df2 = daysss.withColumn("request", lit(1))

#Here I am extracting day from day of the month

dayssssss = df2.withColumn("days",F.dayofmonth(df2['date']))
dayssssss.show()
 
spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")
dayssssss.groupBy("week_day_full").count().na.drop().orderBy('week_day_full').show()
# Using Sum to calculate sum of the request
df5 = dayssssss.groupby('week_day_full','days').sum('request')
df5.show()
df6 = df5.select(['week_day_full','sum(request)'])

df6.show()

df7 = df6.groupby('week_day_full').max().na.drop().alias('Max')
df8 = df6.groupby('week_day_full').min().na.drop().alias('Min')
#df9 = df7.join(df8,df7.week_day_full ==  df8.week_day_full,"inner")
# Here I am combining the output
df9 = df7.join(df8,['week_day_full'])
df9.show()
# Here I am doing visualisation
label = df9.select('week_day_full').rdd.flatMap(lambda x: x).collect()
maximum  = df9.select('max(sum(request))').rdd.flatMap(lambda x: x).collect()
minimum = df9.select('min(sum(request))').rdd.flatMap(lambda x: x).collect()




xx = np.arange(len(label))
width = 0.35
fig,ax = plt.subplots()
first = ax.bar(xx- width/2,minimum,width,label='Minimum',color='red')
second = ax.bar(xx+ width/2,maximum,width,label='Maximum',color='blue')
ax.set_ylabel("No of Requests")
ax.set_title("Maximum and Minimum request by days")
plt.savefig("../Output/Q1B.png")

mpgs = data.select(F.col('request')).where(F.col('request').contains('.mpg')).groupBy('request').count()
sortedmpgs = mpgs.sort('count', ascending = False)

sortedmpgs.show(truncate = False)
import re

from pyspark.sql.functions import split
# Here I am splitting the string to get the desired output
split1 = sortedmpgs.withColumn("col1", split(col("request"), " ").getItem(0)).withColumn("col2", split(col("request"), " ").getItem(1))

split1.show(20,False)
split2 = split1.withColumn("col3", split(col("col2"), "/").getItem(0)).withColumn("col4", split(col("col2"), "/").getItem(1)).withColumn("col5", split(col("col2"), "/").getItem(2)).withColumn("col6", split(col("col2"), "/").getItem(3)).withColumn("col7", split(col("col2"), "/").getItem(4)).withColumn("col8", split(col("col2"), "/").getItem(5))
split2.show()

frame = split2.select(['count','col7'])
frame.show()

new_frame1 = frame.where(frame.col7=='gemini-launch.mpg')
new_frame2 = new_frame1.withColumnRenamed("col7","file_name")
new_frame2.show()

frame2 = split2.select(['count','col8'])
frame2.show()
frame3 =frame2.dropna()
frame3.show()
new_frame3 = frame3.withColumnRenamed("col8","file_name")
new_frame3.show()

final_frame = new_frame3.union(new_frame2)
final_frame.show(truncate=False)

from pyspark.sql.functions import desc

order = final_frame.orderBy(desc("count"))
order1 = order.limit(12)
order1.show()

reverse = final_frame.orderBy("count")
reverse1 = reverse.limit(12)
reverse1.show()

final_result = order1.union(reverse1)
final_result.show()
# Here I am plotting the graph
label_new = final_result.select('file_name').rdd.flatMap(lambda x: x).collect()
max_count = final_result.select('count').rdd.flatMap(lambda x: x).collect()

plt.rcParams["figure.figsize"] = (20,10)
plt.subplots_adjust(bottom=0.6)
plt.xticks(rotation=90)
plt.bar(label_new,max_count, color='Blue')
plt.title("Top and Bottom 12 Request Counts")
plt.savefig("../Output/Q1_figD.png")


