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
import java.util.Iterator;
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
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.matrix.format.MatrixOutputFormat;

/**
 * Sample columns of a matrix
 * 
 * @author myabandeh
 * 
 */
public class SampleColsJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(SampleColsJob.class);

  public static final String SAMPLERATE = "sampleRate";
  public static final String COLS = "matrix.input.cols";

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(SAMPLERATE, "rate", "samperate");
    addOption(COLS, "cols", "cols");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    float sampleRate = Float.parseFloat(getOption(SAMPLERATE));
    int cols = Integer.parseInt(getOption(COLS));
    run(getConf(), getInputPath(), cols, getOutputPath(), sampleRate);
    return 0;
  }

  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix A, float sampleRate, String label)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + SampleColsJob.class.getName());
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    SampleColsJob job = new SampleColsJob();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), A.numCols(), outPath, sampleRate);
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes =
        new DistributedRowMatrix(outPath, A.getOutputTempPath(), A.numRows(),
            A.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  public void run(Configuration conf, Path matrixInputPath, int cols,
      Path matrixOutputPath, float sampleRate) throws IOException,
      InterruptedException, ClassNotFoundException {
    conf = new Configuration(conf);

    conf.setFloat(SAMPLERATE, sampleRate);
    conf.setInt(COLS, cols);
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPath,
        "samplecol");

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(SampleColsJob.class);
    job.setJobName(SampleColsJob.class.getSimpleName() + "-"
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
    private float sampleRate;
    private int cols;
    private float[] chances;
    VectorWritable vw = new VectorWritable();

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      sampleRate = conf.getFloat(SAMPLERATE, 0.01f);
      cols = conf.getInt(COLS, Integer.MAX_VALUE);
      chances = new float[cols];
      Random rand = new Random(0);// all rows start with the same seed
      for (int c = 0; c < cols; c++)
        chances[c] = rand.nextFloat();
    }

    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException, InterruptedException {
      Vector sampledVector = sample(v.get());
      vw.set(sampledVector);
      context.write(r, vw);
    }

    Vector sample(Vector inVec) {
      RandomAccessSparseVector samVec =
          new RandomAccessSparseVector(inVec.size(), (int) (inVec.size()
              * sampleRate * 1.1));
      Iterator<Vector.Element> it = inVec.nonZeroes().iterator();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        int index = e.index();
        if (pass(index))
          samVec.set(index, e.get());
      }
      return new SequentialAccessSparseVector(samVec);
    }

    boolean pass(int index) {
      float selectChance = chances[index];
      return (selectChance <= sampleRate);
    }
  }
}
