# Distributed Big Data Infrastructure Guide
## Apache Spark & Hadoop YARN Integration Setup for Linux Virtual Machine

This document details the complete technical configuration required to run a  distributed machine learning pipeline using Apache Spark, Hadoop HDFS, and YARN within a Linux Virtual Machine.

---

## 1. Install Apache Spark
Download the package using wget from https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz

```bash
wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
```

Extract the downloaded package to `/home/user/spark`:

```bash
tar -xvf spark-3.5.1-bin-hadoop3.tgz -C /home/user/spark
```

## 2. Local Environment Shell Configuration (`~/.bashrc`)
The `.bashrc` script configures the user-space environment variables required to link the decoupled Apache Spark installation with the underlying Hadoop infrastructure files.

Open the configuration file:
```bash
nano ~/.bashrc
```

```
export SPARK_HOME=/home/user/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export YARN_CONF_DIR=$HOME/hadoop/etc/hadoop
```

Run:
```bash
source ~/.bashrc
```

## 3. Create directory in HDFS and import hotel_reservations.csv file 
```bash
hadoop fs -mkdir -p /user/hotel_prediction/data/
hadoop fs -put ~/hotel-reservation/hotel_reservations.csv /user/hotel_prediction/data/
hadoop fs -mkdir -p /user/hotel_prediction/saved_models/
```

## 4. Configure YARN runtime xml

The recommended memory value for YARN is 5GB out of total 8GB VM memory. For CPU core, it is 4 out of total 4 cores.

Stop Hadoop services first:
```bash
./home/user/hadoop/sbin/stop-all.sh
```

Open the configuration file:
```bash
nano ~/hadoop/etc/hadoop/yarn-site.xml
```

Modify or add the following elements inside the primary <configuration> tags:
```xml
<configuration>
    <property>
        <name>yarn.nodemanager.resource.memory-mb</name>
        <value>5120</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.resource.cpu-vcores</name>
        <value>4</value>
    </property>

    <property>
        <name>yarn.scheduler.minimum-allocation-mb</name>
        <value>256</value>
    </property>
    
    <property>
        <name>yarn.scheduler.maximum-allocation-mb</name>
        <value>4096</value>
    </property>

    <property>
        <name>yarn.nodemanager.pmem-check-enabled</name>
        <value>false</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.vmem-check-enabled</name>
        <value>false</value>
    </property>
</configuration>
```

Start Hadoop services again:
```bash
./home/user/hadoop/sbin/start-all.sh
```

## 5. Run training script
```bash
spark-submit \
    --master yarn \
    --deploy-mode client \
    --py-files bigdata.py \
    --driver-memory 1g \
    --executor-memory 1536m \
    --executor-cores 2 \
    --num-executors 2 \
    hotel_prediction.py
```

## 6. Run model inspection script (Optional)
```bash
spark-submit view_model.py
```