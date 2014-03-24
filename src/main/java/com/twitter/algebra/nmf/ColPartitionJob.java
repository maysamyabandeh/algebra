/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Partition matrix A by column. It is necessary when A has many rows (At has
 * many columns). It also allows {@link AtB_DMJ} to split the multiplication
 * into more aggregatable multiplications.
 * 
 * Note: there is a tradeoff for the number of column partitions. Many column
 * partitions hurt the performance by redundant remote reading of the other
 * matrix.
 * 
 * @author myabandeh
 */
public class ColPartitionJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(ColPartitionJob.class);

  public static final String NUM_ORIG_ROWS_KEY = "Matrix.numRows";
  public static final String NUM_ORIG_COLS_KEY = "Matrix.numCols";
  public static final String NUM_COL_PARTITIONS = "Matrix.numColPartitions";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ColPartitionJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    addOption("resColPart", "cp",
        "Number of column partitions of the output matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));
    int numColPartitions = Integer.parseInt(getOption("resColPart"));

    DistributedRowMatrix matrix =
        new DistributedRowMatrix(getInputPath(), getTempPath(), numRows,
            numCols);
    matrix.setConf(new Configuration(getConf()));
    partition(matrix, getConf(), "Transpose-" + getInputPath(),
        numColPartitions);

    return 0;
  }

  /**
   * Partition A by columns. Refer to {@link ColPartitionJob} for further
   * details.
   * 
   * @param distM input matrix A
   * @param conf the initial configuration
   * @param label the label for the output directory
   * @param numColPartitions the hint for the desired number of column
   *          partitions
   * @return Partitioned A wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix partition(DistributedRowMatrix distM,
      Configuration conf, String label, int numColPartitions)
      throws IOException, InterruptedException, ClassNotFoundException {
    Path outputPath = new Path(distM.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outputPath.toUri(), conf);
    if (!fs.exists(outputPath)) {
      ColPartitionJob job = new ColPartitionJob();
      job.run(conf, distM.getRowPath(), outputPath, distM.numRows(),
          distM.numCols(), numColPartitions);
    } else {
      log.warn("----------- Skip already exists: " + outputPath);
    }
    DistributedRowMatrix m =
        new DistributedRowMatrix(outputPath, distM.getOutputTempPath(),
            distM.numRows(), distM.numCols());
    m.setConf(conf);
    return m;
  }

  /**
   * Partition A on columns, where A refers to the path that contain a matrix in
   * {@link SequenceFileInputFormat}. Refer to {@link ColPartitionJob} for
   * further details.
   * 
   * @param conf the initial configuration
   * @param matrixInputPath the path to the input matrix A
   * @param matrixOutputPath the path of the resulting partitioned matrix
   * @param numInputRows rows
   * @param numInputCols cols
   * @param numColPartitions the hint for the desired number of column
   *          partitions
   * @return the running job
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public Job run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath, int numInputRows, int numInputCols,
      int numColPartitions) throws IOException, InterruptedException,
      ClassNotFoundException {
    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "colpartition");

    int colPartSize = getColPartitionSize(numInputCols, numColPartitions);
    numColPartitions = (int) Math.ceil(numInputCols / (double) colPartSize);

    if (numReducers < numColPartitions)
      numReducers = numColPartitions;

    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPath, "colpartition");

    conf.setInt(NUM_ORIG_ROWS_KEY, numInputRows);
    conf.setInt(NUM_ORIG_COLS_KEY, numInputCols);
    conf.setInt(NUM_COL_PARTITIONS, numColPartitions);
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(ColPartitionJob.class);
    job.setJobName(ColPartitionJob.class.getSimpleName());

    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    MultipleInputs.addInputPath(job, matrixInputPath,
        SequenceFileInputFormat.class);

    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(ElementWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    RowColPartitioner.setPartitioner(job,
        RowColPartitioner.ElementRowColPartitioner.class, numInputRows,
        numInputCols, numColPartitions);

    job.setReducerClass(MyReducer.class);
    job.setNumReduceTasks(numReducers);

    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
    return job;
  }

  /**
   * Partition each row to N partitions and send each as a vector to reducers
   * 
   * @author myabandeh
   * 
   */
  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, ElementWritable, VectorWritable> {
    private int numCols;
    private int numColPartitions;
    int colPartSize;
    private ElementWritable ew = new ElementWritable();
    private VectorWritable vw = new VectorWritable();

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      numCols = conf.getInt(NUM_ORIG_COLS_KEY, Integer.MAX_VALUE);
      numColPartitions = conf.getInt(NUM_COL_PARTITIONS, 1);
      colPartSize = getColPartitionSize(numCols, numColPartitions);
    }

    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException, InterruptedException {
      int row = r.get();
      ew.setRow(row);

      Iterator<Vector.Element> it = v.get().nonZeroes().iterator();
      RandomAccessSparseVector slice = null;
      int nextColEnd = colPartSize;
      while (it.hasNext()) {
        Vector.Element e = it.next();
        if (e.index() >= nextColEnd) {
          if (slice != null) {
            ew.setCol(nextColEnd - colPartSize);
            vw.set(new SequentialAccessSparseVector(slice));
            context.write(ew, vw);
            int jump = (e.index() - nextColEnd) / colPartSize * colPartSize;
            nextColEnd += colPartSize;
            nextColEnd += jump;
          }
          slice = null;
        }
        if (slice == null)
          slice = new RandomAccessSparseVector(numCols, colPartSize);
        slice.setQuick(e.index(), e.get());
      }
      if (slice != null) {
        ew.setCol(nextColEnd - colPartSize);
        vw.set(new SequentialAccessSparseVector(slice));
        context.write(ew, vw);
      }
    }

  }

  /**
   * Write out each partition vectors generates by the mappers. Relies on
   * {@link RowColPartitioner} to gather the vectors of the same partition
   * together.
   * 
   * Note: If number of reducers is less than number of column-partitions, some
   * partitions will be mixed.
   * 
   * @author myabandeh
   * 
   */
  public static class MyReducer extends
      Reducer<ElementWritable, VectorWritable, IntWritable, VectorWritable> {
    private IntWritable iw = new IntWritable();
    private VectorWritable vw = new VectorWritable();

    @Override
    public void reduce(ElementWritable key, Iterable<VectorWritable> vectors,
        Context context) throws IOException, InterruptedException {
      Iterator<VectorWritable> it = vectors.iterator();
      if (!it.hasNext())
        return;
      iw.set(key.getRow());
      vw.set(it.next().get());
      context.write(iw, vw);
      if (it.hasNext())
        throw new IOException("Two vectors are assigned to the same key: "
            + key + " first: " + vw.get() + " second: " + it.next());
    }

  }

  public static int getColPartitionSize(int numCols, int numColPartitions) {
    int colPartSize = (int) Math.ceil(numCols / (double) numColPartitions);
    return colPartSize;
  }
}
