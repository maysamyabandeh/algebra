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
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
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

/**
 * Create a distributed martix initialized by random data
 */
public class DistRndMatrixJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(DistRndMatrixJob.class);

  public static final String ROWS = "input.matrix.rows";
  public static final String COLS = "input.matrix.cols";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DistRndMatrixJob(), args);
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

    random(getConf(), numRows, numCols, getTempPath(), "cmdline");

    return 0;
  }

  public static DistributedRowMatrix random(Configuration conf, int rows,
      int cols, Path tmpPath, String label) throws IOException,
      InterruptedException, ClassNotFoundException {
    Path inputPath = new Path(tmpPath, "null-in");
    Path outputPath =
        new Path(tmpPath, "Random-" + label + "-" + rows + "-" + cols);
    FileSystem fs = FileSystem.get(outputPath.toUri(), conf);
    DistRndMatrixJob job = new DistRndMatrixJob();
    if (!fs.exists(inputPath)) {
      FSDataOutputStream inFile = fs.create(inputPath);
      inFile.write("NullValue".getBytes());
      inFile.close();
    }
    if (!fs.exists(outputPath)) {
      job.run(conf, inputPath, outputPath, rows, cols);
    } else {
      log.warn("----------- Skip already exists: " + outputPath);
    }
    DistributedRowMatrix distRes =
        new DistributedRowMatrix(outputPath, tmpPath, rows, cols);
    distRes.setConf(conf);
    return distRes;
  }

  public void run(Configuration conf, Path inPath, Path matrixOutputPath,
      int numInputRows, int numInputCols) throws IOException,
      InterruptedException, ClassNotFoundException {
    conf.setInt(ROWS, numInputRows);
    conf.setInt(COLS, numInputCols);

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(DistRndMatrixJob.class);
    job.setJobName(DistRndMatrixJob.class.getSimpleName() + "-"
        + matrixOutputPath.getName());

    FileSystem fs = FileSystem.get(inPath.toUri(), conf);
    inPath = fs.makeQualified(inPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, inPath);
    job.setInputFormatClass(TextInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(NullWritable.class);

    // ensures total order (when used with {@link MatrixOutputFormat}),
    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        numInputRows);

    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "random");
    job.setNumReduceTasks(numReducers);

    job.setReducerClass(MyReducer.class);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  /**
   * This mappers does nothing but initializing the reducers
   * 
   * @author myabandeh
   * 
   */
  public static class MyMapper extends
      Mapper<LongWritable, Text, IntWritable, NullWritable> {
    private IntWritable iw = new IntWritable();
    private int rows;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      rows = conf.getInt(ROWS, Integer.MAX_VALUE);
    }

    @Override
    public void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException {
      for (int i = 0; i < rows; i++) {
        iw.set(i);
        context.write(iw, NullWritable.get());
      }
    }
  }

  public static class MyReducer
      extends
      Reducer<WritableComparable<?>, NullWritable, WritableComparable<?>, VectorWritable> {
    Random rnd;
    VectorWritable vw = new VectorWritable();
    private int cols;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      cols = conf.getInt(COLS, Integer.MAX_VALUE);
      rnd = new Random(context.getTaskAttemptID().getId());
    }

    @Override
    public void reduce(WritableComparable<?> key, Iterable<NullWritable> nulls,
        Context context) throws IOException, InterruptedException {
      Vector vector = new RandomAccessSparseVector(cols);
      for (int i = 0; i < cols; i++)
        vector.set(i, rnd.nextDouble());
      vw.set(new SequentialAccessSparseVector(vector));
      context.write(key, vw);
    }
  }
}
