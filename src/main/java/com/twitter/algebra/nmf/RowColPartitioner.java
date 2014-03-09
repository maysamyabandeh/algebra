package com.twitter.algebra.nmf;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Partitioner;

/**
 * 
 * An example of how to extend {@link RowColPartitioner} is
 * {@link ElementRowColPartitioner} class.
 * 
 * @author myabandeh
 * 
 * @param <K>
 *          the map output key
 * @param <V>
 *          the map output value
 */
public abstract class RowColPartitioner<K, V> extends Partitioner<K, V> implements
    Configurable {
  public static String TOTAL_ROWS = "rowcolpartitioner.num.rows";
  public static String TOTAL_COLS = "rowcolpartitioner.num.cols";
  public static String TOTAL_COL_PARTITIONS = "rowcolpartitioner.num.col.partitions";
  protected int totalRows = -1;
  protected int totalCols = -1;
  protected int totalColPartitions = -1;
  protected Configuration conf;

  /**
   * Configure a job to use RowPartitioner
   * @param job the job
   * @param pClass the class that inherits RowPartitioner
   * @param totalRows total number of rows
   */
  @SuppressWarnings("rawtypes")
  public static void setPartitioner(Job job, Class<? extends RowColPartitioner> pClass,
      int totalRows, int totalCols, int totalColPartitions) {
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
      throw new IllegalArgumentException("TOTAL_ROWS is not set in RowColPartitioner");
    if (totalCols == -1)
      throw new IllegalArgumentException("TOTAL_COLS is not set in RowColPartitioner");
    if (totalColPartitions == -1)
      throw new IllegalArgumentException("TOTAL_COL_PARTITIONS is not set in RowColPartitioner");
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
  protected int getPartition(int row, int col, int numPartitions) {
//    return (int) ((row / (double) totalRows) * numPartitions);
    int cPart = (int) (col * (long) totalColPartitions / (double) totalCols);
    int totalRowPartitions = numPartitions / totalColPartitions;
    int rPart =  (int) (row * (long) totalRowPartitions / (double) totalRows);
    int part = rPart * totalColPartitions + cPart;
//    if (row==0 && col < 300 && col > 270)
//      System.out.println("#### col" + col + " row " + row + " cPart " + cPart
//          + " rPArt " + rPart + " part: " + part + " toralCols " + totalCols
//          + " totalColPartitions " + totalColPartitions);
    return part;
  }

  /**
   * To use {@link RowColPartitioner}, we need to define how the index extracted
   * from the key. IntRowPartitioner does that for keys of type IntWritable.
   * 
   * @author myabandeh
   * 
   * @param <V>
   */
  public static class ElementRowColPartitioner<V> extends RowColPartitioner<ElementWritable, V> {
    @Override
    public int getPartition(ElementWritable element, V value,
        int numPartitions) {
      return getPartition(element.getRow(), element.getCol(), numPartitions);
    }
  }
}























