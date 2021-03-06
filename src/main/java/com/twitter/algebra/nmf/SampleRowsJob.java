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

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.matrix.format.MatrixOutputFormat;

/**
 * Sample rows of a matrix
 * 
 * @author myabandeh
 * 
 */
public class SampleRowsJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(SampleRowsJob.class);

  public static final String SAMPLERATE = "sampleRate";

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(SAMPLERATE, "rate", "samperate");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    float sampleRate = Float.parseFloat(getOption(SAMPLERATE));
    run(getConf(), getInputPath(), getOutputPath(), sampleRate);
    return 0;
  }

  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix A, float sampleRate, String label)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + SampleRowsJob.class.getName());
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    SampleRowsJob job = new SampleRowsJob();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), outPath, sampleRate);
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
      Path matrixOutputPath, float sampleRate) throws IOException,
      InterruptedException, ClassNotFoundException {
    conf = new Configuration(conf);

    conf.setFloat(SAMPLERATE, sampleRate);
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPath,
        "samplerows");

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(SampleRowsJob.class);
    job.setJobName(SampleRowsJob.class.getSimpleName() + "-"
        + matrixOutputPath.getName());

    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);

    job.setNumReduceTasks(0);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    Random rand = new Random(0);// all mappers start with the same seed
    private float sampleRate;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      sampleRate = conf.getFloat(SAMPLERATE, 0.01f);
    }

    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException, InterruptedException {
      if (pass(r.get()))
        context.write(r, v);
    }

    int lastIndex = -1;

    boolean pass(int index) {
      float selectChance = 0;
      // TODO: assume that lastIndex < index
      while (lastIndex < index) {
        lastIndex++;
        selectChance = rand.nextFloat();
      }
      // lastIndex = index
      return (selectChance <= sampleRate);
    }

  }

}
