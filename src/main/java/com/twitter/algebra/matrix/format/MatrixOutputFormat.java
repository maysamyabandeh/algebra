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

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.OutputCommitter;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MapFileOutputFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * We enforce total order in MapReudce output via {@link RowPartitioner} and
 * {@link MatrixOutputFormat}. Each matrix job needs to derive its partitioner
 * from {@link RowPartitioner} and set its output format from
 * {@link MatrixOutputFormat}. The output of each reducer must be also locally
 * sorted.
 * 
 * {@link MatrixOutputFormat} uses MapFile for its output. But also it tagged
 * the MapFile name with the first key (smallest key) inserted into the MapFile.
 * This allows the output of the job to be efficiently retrieved via
 * {@link MapDir} class.
 * 
 * @author myabandeh
 * 
 */
public class MatrixOutputFormat extends
    LazyOutputFormat<WritableComparable<?>, Writable> {
  private static final Logger log = LoggerFactory
      .getLogger(MatrixOutputFormat.class);

  //TODO: does it have to be static?
  //allow to cover both IntWritable and LongWritable
  private static Long firstKey = null;

  private static void setFirstKey(long key) {
    if (firstKey == null)
      firstKey = new Long(key);
  }

  private static void setFirstKey(Object key) {
    if (key instanceof IntWritable)
      setFirstKey(((IntWritable)key).get());
    if (key instanceof LongWritable)
      setFirstKey(((LongWritable)key).get());
    //else throws exception when creating the file name
  }

  private void getBaseOutputFormat(Configuration conf) throws IOException {
    baseOut = new MatrixMapFileOutputFormat();
  }

  @Override
  public RecordWriter<WritableComparable<?>, Writable> getRecordWriter(
      TaskAttemptContext context) throws IOException, InterruptedException {
    if (baseOut == null) {
      getBaseOutputFormat(context.getConfiguration());
    }
    return new LazyRecordWriter<WritableComparable<?>, Writable>(baseOut,
        context);
  }

  @Override
  public void checkOutputSpecs(JobContext context) throws IOException,
      InterruptedException {
    if (baseOut == null) {
      getBaseOutputFormat(context.getConfiguration());
    }
    super.checkOutputSpecs(context);
  }

  @Override
  public OutputCommitter getOutputCommitter(TaskAttemptContext context)
      throws IOException, InterruptedException {
    if (baseOut == null) {
      getBaseOutputFormat(context.getConfiguration());
    }
    return super.getOutputCommitter(context);
  }

  /**
   * We need a LazyRecordWriter to create the output MapDirs only after the
   * first insertion. This is because we need the first inserted key to name the
   * MapDir.
   */
  private static class LazyRecordWriter<K, V> extends FilterRecordWriter<K, V> {
    final OutputFormat<K, V> outputFormat;
    final TaskAttemptContext taskContext;

    public LazyRecordWriter(OutputFormat<K, V> out,
        TaskAttemptContext taskContext) throws IOException,
        InterruptedException {
      this.outputFormat = out;
      this.taskContext = taskContext;
    }

    @Override
    public void write(K key, V value) throws IOException, InterruptedException {
      if (rawWriter == null) {
        setFirstKey(key);
        rawWriter = outputFormat.getRecordWriter(taskContext);
      }
      try {
        rawWriter.write(key, value);
      } catch (IOException e) {
        //TODO: I get this error with some data set as there are a few out-of-order
        //records.
        log.info("out of order key? " + e.getMessage());
        e.printStackTrace();
      }
    }

    @Override
    public void close(TaskAttemptContext context) throws IOException,
        InterruptedException {
      if (rawWriter != null) {
        rawWriter.close(context);
      }
       //Important if the JVM is reused, the static variable should be reset
      firstKey = null;
    }
  }

  public static class MatrixMapFileOutputFormat extends MapFileOutputFormat {
    String getUniqueFile() throws IOException {
      if (firstKey == null)
        throw new IOException("the first key is not set for "
            + MatrixMapFileOutputFormat.class);
      return "matrix-k-" + firstKey;
    }

    @Override
    public Path getDefaultWorkFile(TaskAttemptContext context, String extension)
        throws IOException {
      FileOutputCommitter committer = (FileOutputCommitter) getOutputCommitter(context);
      return new Path(committer.getWorkPath(), getUniqueFile());
    }
  }

}
