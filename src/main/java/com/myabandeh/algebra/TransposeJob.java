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

package com.myabandeh.algebra;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;
import com.twitter.algebra.nmf.NMFCommon;

/**
 * Transpose a matrix borrowed from Mahout's {@link TransposeJob}
 */
public class TransposeJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(TransposeJob.class);

  public static final String NUM_ORIG_ROWS_KEY = "SparseRowMatrix.numRows";
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new TransposeJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));

    DistributedRowMatrix matrix = new DistributedRowMatrix(getInputPath(),
        getTempPath(), numRows, numCols);
    matrix.setConf(new Configuration(getConf()));
    transpose(matrix, getConf(), "Transpose-" + getInputPath());

    return 0;
  }

  /**
   * Perform transpose of A, where A is already wrapped in a DistributedRowMatrix
   * object. 
   * 
   * @param distM
   *          input matrix A
   * @param conf
   *          the initial configuration
   * @param label
   *          the label for the output directory
   * @return At wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix transpose(DistributedRowMatrix distM,
      Configuration conf, String label) throws IOException, InterruptedException, ClassNotFoundException {
    Path outputPath = new Path(distM.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outputPath.toUri(), conf);
    TransposeJob job = new TransposeJob();
    if (!fs.exists(outputPath)) {
      job.run(conf, distM.getRowPath(), outputPath, distM.numRows(), distM.numCols());
    } else {
      log.warn("----------- Skip already exists: " + outputPath);
    }
    DistributedRowMatrix m = new DistributedRowMatrix(outputPath,
        distM.getOutputTempPath(), distM.numCols(), distM.numRows());
    m.setConf(conf);
    return m;
  }

  /**
   * Perform transpose of A, where A refers to the path that contains a matrix
   * in {@link SequenceFileInputFormat}.
   * 
   * @param conf
   *          the initial configuration
   * @param matrixInputPath
   *          the path to the input files that we process
   * @param matrixOutputPath
   *          the path of the resulting transpose matrix
   * @param numInputRows
   *          rows
   * @param numInputCols
   *          cols
   * @return the running job
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath, int numInputRows, int numInputCols) throws IOException, InterruptedException, ClassNotFoundException {
    conf.setInt(NUM_ORIG_ROWS_KEY, numInputRows);
    conf.setInt(RowPartitioner.TOTAL_KEYS, numInputCols);
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, new Path[] {matrixInputPath}, "transpose");
    
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(TransposeJob.class);
    job.setJobName(TransposeJob.class.getSimpleName());

    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(TransposeMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);
    
    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "transpose");
    job.setNumReduceTasks(numReducers);
//    job.setPartitionerClass(RowPartitioner.IntRowPartitioner.class);
    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        numInputCols);
    job.setCombinerClass(MergeVectorsCombiner.class);
    job.setReducerClass(MergeVectorsReducer.class);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  public static class TransposeMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private int newNumCols;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      newNumCols = conf.getInt(NUM_ORIG_ROWS_KEY, Integer.MAX_VALUE);
    }

    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException, InterruptedException {
      int row = r.get();
      Iterator<Vector.Element> it = v.get().nonZeroes().iterator();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        RandomAccessSparseVector tmp = new RandomAccessSparseVector(newNumCols,
            1);
        tmp.setQuick(row, e.get());
        r.set(e.index());
        context.write(r, new VectorWritable(tmp));
      }
    }
  }

  public static class MergeVectorsCombiner
      extends
      Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

    @Override
    public void reduce(WritableComparable<?> key,
        Iterable<VectorWritable> vectors, Context context) throws IOException,
        InterruptedException {
      context.write(key, VectorWritable.merge(vectors.iterator()));
    }
  }

  public static class MergeVectorsReducer
      extends
      Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {
    @Override
    public void reduce(WritableComparable<?> key,
        Iterable<VectorWritable> vectors, Context context) throws IOException,
        InterruptedException {
      Vector merged = VectorWritable.merge(vectors.iterator()).get();
      context.write(key, new VectorWritable(new SequentialAccessSparseVector(
          merged)));
    }
  }
}
