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

package com.twitter.algebra.matrix.multiply;

import java.io.IOException;
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
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.MergeVectorsReducer;
import com.twitter.algebra.matrix.format.MatrixOutputFormat;
import com.twitter.algebra.matrix.format.RowPartitioner;

/**
 * Partition a row matrix into fixed number of partitions to be used by map-side
 * join
 * 
 * @author myabandeh
 * 
 */
public class PartitionerJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(PartitionerJob.class);

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    addOption("numParts", "np", "Number of partitions of the input matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));
    int numParts = Integer.parseInt(getOption("numParts"));

    DistributedRowMatrix matrix =
        new DistributedRowMatrix(getInputPath(), getTempPath(), numRows,
            numCols);
    matrix.setConf(new Configuration(getConf()));
    PartitionerJob.run(getConf(), matrix, numParts, "partitioned-" + getInputPath());
    return 0;
  }

  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix A, int partitions, String label) throws IOException,
      InterruptedException, ClassNotFoundException {
    log.info("running " + PartitionerJob.class.getName());
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    PartitionerJob job = new PartitionerJob();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), outPath, A.numRows(), partitions);
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes =
        new DistributedRowMatrix(outPath, A.getOutputTempPath(), A.numRows(),
            A.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  public void run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath, int aRows, int partitions) throws IOException,
      InterruptedException, ClassNotFoundException {
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(PartitionerJob.class);
    job.setJobName(PartitionerJob.class.getSimpleName() + "-"
        + matrixOutputPath.getName());
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);

    job.setNumReduceTasks(partitions);

    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(IdMapper.class);
    job.setReducerClass(IdReducer.class);

    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        aRows);

    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  public static class IdMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    @Override
    public void map(IntWritable key, VectorWritable value, Context context)
        throws IOException, InterruptedException {
      context.write(key, value);
    }
  }

  public static class IdReducer
      extends
      Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {
    @Override
    public void reduce(WritableComparable<?> key,
        Iterable<VectorWritable> vectors, Context context) throws IOException,
        InterruptedException {
      context.write(key, vectors.iterator().next());
    }
  }

}
