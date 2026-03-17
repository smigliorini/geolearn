val staion_clusters = spark.read.option("header", true).csv("./weather_data/station-clusters.csv")

staion_clusters.printSchema
staion_clusters.createOrReplaceTempView("stations")


val daily_weather = spark.read.option("header", true).csv("./weather_data/data/").withColumn("PARSED_DATE", to_date(col("DATE"),"yyyy-MM-dd"))

daily_weather.printSchema

daily_weather.createOrReplaceTempView("daily")

val df = spark.sql("SELECT STATION, year(PARSED_DATE) AS YEAR, month(PARSED_DATE) AS MONTH, FIRST(LONGITUDE) AS LONGITUDE, FIRST(LATITUDE) AS LATITUDE, FIRST(ELEVATION) AS ELEVATION, AVG(TMAX) AS TEMP FROM daily WHERE TMAX < 700 AND TMAX > -700 GROUP BY STATION, YEAR, MONTH")
df.repartition(1).write.partitionBy("YEAR", "MONTH").option("header",true).mode("overwrite").csv("./weather_data/monthly_partitions/")
