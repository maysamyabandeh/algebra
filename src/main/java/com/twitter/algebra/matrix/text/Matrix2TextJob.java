package com.twitter.algebra.matrix.text;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Matrix2TextJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(Matrix2TextJob.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Matrix2TextJob(), args);
  }
  
  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    run(getConf(), getInputPath(), new Path(getOutputPath(),  "text-" + getInputPath().getName()));
    return 0;
  }

  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix A, String label)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + Matrix2TextJob.class.getName());

    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    Matrix2TextJob job = new Matrix2TextJob();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), outPath);
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes = new DistributedRowMatrix(outPath,
        A.getOutputTempPath(), A.numRows(), A.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  public void run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath) throws IOException, InterruptedException,
      ClassNotFoundException {
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(Matrix2TextJob.class);
    job.setJobName(Matrix2TextJob.class.getSimpleName());
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

//    FileInputFormat.addInputPath(job, matrixInputPath);
    MultipleInputs.addInputPath(job, matrixInputPath, SequenceFileInputFormat.class);
//    job.setInputFormatClass(SequenceFileInputFormat.class);
    TextOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setNumReduceTasks(0);

    job.setOutputFormatClass(TextOutputFormat.class);
//    job.setOutputKeyClass(IntWritable.class);
//    job.setOutputValueClass(org.apache.hadoop.io.Text);
    job.setMapperClass(IdMapper.class);

    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }
  
  static double EPSILON = Double.NaN;
  static final String EPSILON_STR = "matrix.text.epsilon";

  public static class IdMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    VectorWritable vw = new VectorWritable();
    
    @Override
    public void setup(Context context) throws IOException {
      EPSILON = context.getConfiguration().getDouble(EPSILON_STR, Double.NaN);
    }
    
    @Override
    public void map(IntWritable key, VectorWritable value, Context context) throws IOException,
        InterruptedException {
      RandomAccessSparseVector tmp = new RandomAccessSparseVector(value.get().size(),
          value.get().getNumNondefaultElements());
      for (Element e : value.get().nonZeroes()) {
        if (!Double.isNaN(EPSILON) && e.get() > EPSILON)
          tmp.set(e.index(), e.get());
      }
      vw.set(tmp);
      context.write(key, vw);
    }
  }

}





