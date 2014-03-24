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
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reindex a matrix in edge format to allow the row ids fit into the integer
 * range. The output is the reindex map.
 */
public class ReindexerJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(ReindexerJob.class);
  public final static String TOTALINDEX_COUNTER_GROUP = "Result";
  public final static String TOTALINDEX_COUNTER_NAME = "totalIndex";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ReindexerJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }
    index(getConf(), getInputPath(), getTempPath(), getName(getInputPath()
        .getName()));
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
      Job mrJob = job.run(conf, input, outputPath);
      long totalIndex =
          mrJob.getCounters().getGroup(TOTALINDEX_COUNTER_GROUP)
              .findCounter(TOTALINDEX_COUNTER_NAME).getValue();
      return (int) totalIndex;
    } else {
      log.warn("----------- Skip already exists: " + outputPath);
      return -1;
    }
  }

  public Job run(Configuration conf, Path matrixInputPath, Path matrixOutputPath)
      throws IOException, InterruptedException, ClassNotFoundException {
    conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator",
        "\t");

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(ReindexerJob.class);
    job.setJobName(ReindexerJob.class.getSimpleName() + "-"
        + matrixOutputPath.getName());

    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(KeyValueTextInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(NullWritable.class);

    job.setReducerClass(MyReducer.class);
    // this makes the reindexing very slow but is necessary to have total order
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
    public void map(Text key, Text value, Context context) throws IOException,
        InterruptedException {
      long lrow = Long.parseLong(key.toString());
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
      context.getCounter(TOTALINDEX_COUNTER_GROUP, TOTALINDEX_COUNTER_NAME)
          .increment(nextIndex);
    }

  }
}
