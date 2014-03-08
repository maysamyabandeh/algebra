package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.TransposeJob;
import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;
import com.twitter.algebra.nmf.CompositeDMJ.MyMapper;


public class CombinerJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(CombinerJob.class);

  public static final String SAMPLERATE = "sampleRate";

  @Override
  public int run(String[] strings) throws Exception {

    return 0;
  }

  public static DistributedRowMatrix run(Configuration conf, DistributedRowMatrix A, String label)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + CombinerJob.class.getName());
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    CombinerJob job = new CombinerJob();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), outPath, A.numRows());
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes = new DistributedRowMatrix(outPath,
        A.getOutputTempPath(), A.numRows(), A.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  public void run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath, int aRows) throws IOException, InterruptedException,
      ClassNotFoundException {
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(CombinerJob.class);
    job.setJobName(CombinerJob.class.getSimpleName() + "-" + matrixOutputPath.getName());
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
  //  job.setMapperClass(MyMapper.class);
    
    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "combiner");
    job.setNumReduceTasks(numReducers);// TODO: make it a parameter
    
//    job.setNumReduceTasks(15);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    
    job.setMapperClass(IdMapper.class);
    job.setReducerClass(TransposeJob.MergeVectorsReducer.class);

    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        aRows);

    // since we do not use reducer, to get total order, the map output files has
    // to be renamed after this function returns: {@link AlgebraCommon#fixPartitioningProblem}
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  

  public static class IdMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    @Override
    public void map(IntWritable key, VectorWritable value, Context context) throws IOException,
        InterruptedException {
      context.write(key, value);
    }
  }


}

























