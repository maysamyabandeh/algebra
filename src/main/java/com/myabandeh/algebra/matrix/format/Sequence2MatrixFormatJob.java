package com.myabandeh.algebra.matrix.format;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.nmf.NMFCommon;

/**
 * {@link AtBOuterDynamicMapsideJoin} requires the smaller matrix to be in
 * {@link MapDir} format. By using {@link RowPartitioner} and
 * {@link MatrixOutputFormat}, this is the case for output of the jobs in this
 * package. If the input matrix is not generated by our jobs and hence in
 * {@link SequenceFile} format, this class can convert the set of
 * {@link SequenceFile}s to a {@link MapDir}, given the range of keys covered by
 * each sequence file is disjoint from one another.
 */
public class Sequence2MatrixFormatJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(Sequence2MatrixFormatJob.class);

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    run(getConf(), getInputPath(), getOutputPath());
    return 0;
  }

  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix A, String label)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + Sequence2MatrixFormatJob.class.getName());

    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    Sequence2MatrixFormatJob job = new Sequence2MatrixFormatJob();
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
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPath, "seq2mtx");
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(Sequence2MatrixFormatJob.class);
    job.setJobName(Sequence2MatrixFormatJob.class.getSimpleName());

    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setNumReduceTasks(0);

    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

}