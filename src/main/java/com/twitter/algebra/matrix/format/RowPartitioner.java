/*
Copyright 2014 Twitter, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package com.twitter.algebra.matrix.format;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Partitioner;

/**
 * We enforce total order in MapReudce output via {@link RowPartitioner} and
 * {@link MatrixOutputFormat}. Each matrix job needs to derive its partitioner
 * from {@link RowPartitioner} and set its output format from
 * {@link MatrixOutputFormat}. The output of each reducer must be also locally
 * sorted.
 * 
 * An example of how to extend {@link RowPartitioner} is
 * {@link IntRowPartitioner} class.
 * 
 * @author myabandeh
 * 
 * @param <K>
 *          the map output key
 * @param <V>
 *          the map output value
 */
public abstract class RowPartitioner<K, V> extends Partitioner<K, V> implements
    Configurable {
  public static String TOTAL_KEYS = "num.keys";
  protected int totalKeys = -1;
  protected Configuration conf;

  /**
   * Configure a job to use RowPartitioner
   * @param job the job
   * @param pClass the class that inherits RowPartitioner
   * @param totalKeys total number of rows
   */
  @SuppressWarnings("rawtypes")
  public static void setPartitioner(Job job, Class<? extends RowPartitioner> pClass,
      int totalKeys) {
    job.setPartitionerClass(pClass);
    job.getConfiguration().setInt(TOTAL_KEYS, totalKeys);
  }

  @Override
  public void setConf(Configuration conf) {
    this.conf = conf;
    totalKeys = conf.getInt(TOTAL_KEYS, -1);
    checkTotalKeys();
  }

  protected void checkTotalKeys() {
    if (totalKeys == -1)
      throw new IllegalArgumentException("TOTAL_KEYS is not set in RowPartitioner");
  }

  @Override
  public Configuration getConf() {
    return conf;
  }
  
  /**
   * Assuming the range of rows are in uniformly distributed in [0,totalKeys]
   * (which is the case for matrices) gives balanced partitions that also
   * respect the total order.
   * 
   * @param row
   * @param numPartitions
   * @return
   */
  protected int getPartition(int row, int numPartitions) {
    return (int) ((row * (long) numPartitions / (double) totalKeys));
  }

  /**
   * To use {@link RowPartitioner}, we need to define how the index extracted
   * from the key. IntRowPartitioner does that for keys of type IntWritable.
   * 
   * @author myabandeh
   * 
   * @param <V>
   */
  public static class IntRowPartitioner<V> extends RowPartitioner<IntWritable, V> {
    @Override
    public int getPartition(IntWritable key, V value,
        int numPartitions) {
      return getPartition(key.get(), numPartitions);
    }
  }
}