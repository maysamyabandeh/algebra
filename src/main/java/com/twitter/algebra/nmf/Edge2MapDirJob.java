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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;

/**
 * Convert the edge format to Matrix format. It also reindexes the row ids to
 * fit them into integer range
 */
public class Edge2MapDirJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(Edge2MapDirJob.class);

  public static final String INDEXNAME = "indexName";
  public static final String ROWS = "input.matrix.rows";
  public static final String COLS = "input.matrix.cols";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Edge2MapDirJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    format(getConf(), getInputPath(), getTempPath(), "Format-"
        + getInputPath().getName(), getInputPath().getName());

    return 0;
  }

  public static void format(Configuration conf, Path input, Path tmpPath,
      String label, String name) throws IOException, InterruptedException,
      ClassNotFoundException {
    int totalIndex =
        ReindexerJob.index(conf, input, tmpPath, ReindexerJob.getName(name));
    Path indexPath = new Path(tmpPath, ReindexerJob.getName(name));
    // TODO: here we assume that input matrix is square
    Path outputPath = new Path(tmpPath, label);
    FileSystem fs = FileSystem.get(outputPath.toUri(), conf);
    Edge2MapDirJob job = new Edge2MapDirJob();
    if (!fs.exists(outputPath)) {
      job.run(conf, input, outputPath, totalIndex, totalIndex,
          indexPath.toString());
    } else {
      log.warn("----------- Skip already exists: " + outputPath);
    }
  }

  public void run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath, int numInputRows, int numInputCols, String name)
      throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(INDEXNAME, name);
    conf.setInt(ROWS, numInputRows);
    conf.setInt(COLS, numInputCols);
    conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator",
        "\t");
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPath, "edge2matrix");

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(Edge2MapDirJob.class);
    job.setJobName(Edge2MapDirJob.class.getSimpleName() + "-"
        + matrixOutputPath.getName());

    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(KeyValueTextInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "edge2matrix");
    job.setNumReduceTasks(numReducers);
    // ensures total order (when used with {@link MatrixOutputFormat}),
    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        numInputRows);

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

  public static class MyMapper extends
      Mapper<Text, Text, IntWritable, VectorWritable> {
    private IntWritable iw = new IntWritable();
    private VectorWritable vw = new VectorWritable();
    private HashMap<Long, Integer> hashMap;
    private int cols;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      String name = conf.get(INDEXNAME);
      cols = conf.getInt(COLS, Integer.MAX_VALUE);
      // newNumCols = conf.getInt(NUM_ORIG_ROWS_KEY, Integer.MAX_VALUE);
      hashMap = NMFCommon.readHashMap(name);
    }

    @Override
    public void map(Text key, Text value, Context context) throws IOException,
        InterruptedException {
      long lrow = Long.parseLong(key.toString());
      if (lrow > Integer.MAX_VALUE) {
        context.getCounter("InvalidInput", "toolongkey").increment(1);
        return;
      }
      long lcol = Long.parseLong(value.toString());
      if (lcol > Integer.MAX_VALUE) {
        context.getCounter("InvalidInput", "toolongvalue").increment(1);
        return;
      }
      int row = hashMap.get(lrow);
      int col = hashMap.get(lcol);
      RandomAccessSparseVector vector = new RandomAccessSparseVector(cols, 1);
      vector.setQuick(col, 1);
      iw.set(row);
      vw.set(vector);
      context.write(iw, vw);
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
