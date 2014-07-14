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

package com.twitter.algebra.nmf;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Partitioner;

/**
 * An example of how to extend {@link RowColPartitioner} is
 * {@link ElementRowColPartitioner} class.
 * 
 * @author myabandeh
 * 
 * @param <K> the map output key
 * @param <V> the map output value
 */
public abstract class RowColPartitioner<K, V> extends Partitioner<K, V>
    implements Configurable {
  public static String TOTAL_ROWS = "rowcolpartitioner.num.rows";
  public static String TOTAL_COLS = "rowcolpartitioner.num.cols";
  public static String TOTAL_COL_PARTITIONS =
      "rowcolpartitioner.num.col.partitions";
  protected int totalRows = -1;
  protected int totalCols = -1;
  protected int totalColPartitions = -1;
  protected Configuration conf;

  /**
   * Configure a job to use {@link RowColPartitioner}
   * 
   * @param job the job
   * @param pClass the class that inherits RowPartitioner
   * @param totalRows total number of rows
   */
  @SuppressWarnings("rawtypes")
  public static void setPartitioner(Job job,
      Class<? extends RowColPartitioner> pClass, int totalRows, int totalCols,
      int totalColPartitions) {
    job.setPartitionerClass(pClass);
    job.getConfiguration().setInt(TOTAL_ROWS, totalRows);
    job.getConfiguration().setInt(TOTAL_COLS, totalCols);
    job.getConfiguration().setInt(TOTAL_COL_PARTITIONS, totalColPartitions);
  }

  @Override
  public void setConf(Configuration conf) {
    this.conf = conf;
    totalRows = conf.getInt(TOTAL_ROWS, -1);
    totalCols = conf.getInt(TOTAL_COLS, -1);
    totalColPartitions = conf.getInt(TOTAL_COL_PARTITIONS, -1);
    checkTotalKeys();
  }

  protected void checkTotalKeys() {
    if (totalRows == -1)
      throw new IllegalArgumentException(
          "TOTAL_ROWS is not set in RowColPartitioner");
    if (totalCols == -1)
      throw new IllegalArgumentException(
          "TOTAL_COLS is not set in RowColPartitioner");
    if (totalColPartitions == -1)
      throw new IllegalArgumentException(
          "TOTAL_COL_PARTITIONS is not set in RowColPartitioner");
  }

  @Override
  public Configuration getConf() {
    return conf;
  }

  /**
   * Assuming the range of rows are in uniformly distributed in [0,totalRows]
   * and columns in [0,totalCols] (which is the case for matrices) gives
   * balanced partitions that also respect the total order.
   * 
   * @param row
   * @param col
   * @param numPartitions
   * @return
   */
  protected int getPartition(int row, int col, int numPartitions) {
    int cPart = (int) (col * (long) totalColPartitions / (double) totalCols);
    int totalRowPartitions = numPartitions / totalColPartitions;
    int rPart = (int) (row * (long) totalRowPartitions / (double) totalRows);
    int part = rPart * totalColPartitions + cPart;
    return part;
  }

  /**
   * To use {@link RowColPartitioner}, we need to define how the index extracted
   * from the key. {@link ElementRowColPartitioner} does that for keys of type
   * {@link ElementWritable}.
   * 
   * @author myabandeh
   * 
   * @param <V>
   */
  public static class ElementRowColPartitioner<V> extends
      RowColPartitioner<ElementWritable, V> {
    @Override
    public int getPartition(ElementWritable element, V value, int numPartitions) {
      return getPartition(element.getRow(), element.getCol(), numPartitions);
    }
  }
}
