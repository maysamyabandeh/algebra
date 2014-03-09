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
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
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

import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;

/**
 */
public class ReindexerJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(ReindexerJob.class);
  public final static String TOTALINDEX_COUNTER_GROUP = "Result";
  public final static String TOTALINDEX_COUNTER_NAME = "totalIndex";


  public static final String NUM_ORIG_ROWS_KEY = "SparseRowMatrix.numRows";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ReindexerJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    // addOption("numRows", "nr", "Number of rows of the input matrix");
    // addOption("numCols", "nc", "Number of columns of the input matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    // int numRows = Integer.parseInt(getOption("numRows"));
    // int numCols = Integer.parseInt(getOption("numCols"));

    index(getConf(), getInputPath(), getTempPath(), getName(getInputPath().getName()));

    return 0;
  }
  
  public static String getName(String label) {
    return "Reindex-" + label;
  }

  public static int index(Configuration conf, Path input, Path tmpPath,
      String label) throws IOException, InterruptedException,
      ClassNotFoundException {
    Path outputPath = new Path(tmpPath, label);
    FileSystem fs = FileSystem.get(outputPath.toUri(), conf);
    ReindexerJob job = new ReindexerJob();
    if (!fs.exists(outputPath)) {
      Job mrJob = job.run(conf, input, outputPath, 0, 0);
      long totalIndex = mrJob.getCounters().getGroup(TOTALINDEX_COUNTER_GROUP).findCounter(TOTALINDEX_COUNTER_NAME).getValue();
      return (int)totalIndex;
    } else {
      log.warn("----------- Skip already exists: " + outputPath);
      return -1;
    }
  }

  /**
   * Perform transpose of A, where A refers to the path that contains a matrix
   * in {@link SequenceFileInputFormat}.
   * 
   * @param conf the initial configuration
   * @param matrixInputPath the path to the input files that we process
   * @param matrixOutputPath the path of the resulting transpose matrix
   * @param numInputRows rows
   * @param numInputCols cols
   * @return the running job
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public Job run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath, int numInputRows, int numInputCols)
      throws IOException, InterruptedException, ClassNotFoundException {
    conf.setInt(NUM_ORIG_ROWS_KEY, numInputRows);
    conf.setInt(RowPartitioner.TOTAL_KEYS, numInputCols);
    conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator",
        "\t");

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(ReindexerJob.class);
    job.setJobName(ReindexerJob.class.getSimpleName() + "-" + matrixOutputPath.getName());

    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(KeyValueTextInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(NullWritable.class);

//    job.setPartitionerClass(RowPartitioner.IntRowPartitioner.class);
    // ensures total order (when used with {@link MatrixOutputFormat}),
//    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
//        numCols);

    job.setReducerClass(MyReducer.class);
    
    job.setNumReduceTasks(1);
    
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(IntWritable.class);
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
    return job;
  }

  public static class MyMapper extends
      Mapper<Text, Text, LongWritable, NullWritable> {
    private NullWritable nw = NullWritable.get();
    private LongWritable lw = new LongWritable();

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      // newNumCols = conf.getInt(NUM_ORIG_ROWS_KEY, Integer.MAX_VALUE);
    }

    @Override
    public void map(Text key, Text value, Context context) throws IOException,
        InterruptedException {
      long lrow = Long.parseLong(key.toString());
      // lrow -= Integer.MAX_VALUE;
      if (lrow > Long.MAX_VALUE) {
        context.getCounter("InvalidInput", "toolongkey").increment(1);
        return;
      }
      // throw new IOException("number " + key +
      // " does not fit into an integer");

//      int row = (int) lrow;

      lw.set(lrow);
      context.write(lw, nw);
    }
  }


  public static class MyReducer
      extends
      Reducer<WritableComparable<?>, NullWritable, WritableComparable<?>, WritableComparable<?>> {
    int nextIndex = 0;
    IntWritable iw = new IntWritable();

    @Override
    public void reduce(WritableComparable<?> key,
        Iterable<NullWritable> vectors, Context context) throws IOException,
        InterruptedException {
      iw.set(nextIndex);
      nextIndex++;
      context.write(key, iw);
    }
    
    @Override
    public void cleanup(Context context) throws IOException {
      context.getCounter(TOTALINDEX_COUNTER_GROUP, TOTALINDEX_COUNTER_NAME).increment(nextIndex);
    }
    
  }
}

































