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
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;


public class SampleRowsJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(SampleRowsJob.class);

  public static final String SAMPLERATE = "sampleRate";

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(SAMPLERATE, "rate",
        "samperate");
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
    DistributedRowMatrix distRes = new DistributedRowMatrix(outPath,
        A.getOutputTempPath(), A.numRows(), A.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  public void run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath,float sampleRate) throws IOException, InterruptedException,
      ClassNotFoundException {
    conf.setFloat(SAMPLERATE, sampleRate);
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(SampleRowsJob.class);
    job.setJobName(SampleRowsJob.class.getSimpleName() + "-" + matrixOutputPath.getName());
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);

    NMFCommon.setNumberOfMapSlots(conf, fs, new Path[] {matrixInputPath}, "samplerows");

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

    // since we do not use reducer, to get total order, the map output files has
    // to be renamed after this function returns: {@link AlgebraCommon#fixPartitioningProblem}
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    Random rand = new Random(0);//all mappers start with the same seed
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
      //TODO: assume that lastIndex < index
      while (lastIndex < index) {
        lastIndex++;
        selectChance = rand.nextFloat();
      }
      //lastIndex = index
      return (selectChance <= sampleRate);
    }

  }
  

}

























