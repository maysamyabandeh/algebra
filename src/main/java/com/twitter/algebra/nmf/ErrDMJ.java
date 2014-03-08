package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.AlgebraCommon;
import com.myabandeh.algebra.AlgebraCommon.ExcludeMetaFilesFilter;
import com.myabandeh.algebra.matrix.format.MapDir;
import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;
import com.myabandeh.algebra.matrix.format.RowPartitioner.IntRowPartitioner;

/**
 * | X - A * Y |
 */
public class ErrDMJ extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(ErrDMJ.class);

  public static final String MAPDIRMATRIXX = "mapDirMatrixX";
  public static final String MAPDIRMATRIXYT = "mapDirMatrixYt";
  public static final String YTROWS = "YtRows";
  public static final String YTCOLS = "YtCols";
  
  public static void run(Configuration conf,
      DistributedRowMatrix X, DistributedRowMatrix A, DistributedRowMatrix Yt,   String label) throws IOException, InterruptedException,
      ClassNotFoundException {
    log.info("running " + ErrDMJ.class.getName());
    if (X.numRows() != A.numRows()) {
      throw new CardinalityException(A.numRows(), A.numRows());
    }
    if (A.numCols() != Yt.numCols()) {
      throw new CardinalityException(A.numCols(), Yt.numCols());
    }
    if (X.numCols() != Yt.numRows()) {
      throw new CardinalityException(X.numCols(), Yt.numRows());
    }
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    ErrDMJ job = new ErrDMJ();
    if (!fs.exists(outPath)) {
      job.run(conf, X.getRowPath(), new Path [] {A.getRowPath()}, Yt.getRowPath(), outPath, A.numRows(), Yt.numRows(), Yt.numCols());
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
  }

  public Job run(Configuration conf, Path xPath, Path[] matrixAInputPaths,
      Path ytPath, Path outPath, int aRows, int ytRows, int ytCols) throws IOException,
      InterruptedException, ClassNotFoundException {

    conf.set(MAPDIRMATRIXX, xPath.toString());
    conf.set(MAPDIRMATRIXYT, ytPath.toString());
    conf.setInt(YTROWS, ytRows);
    conf.setInt(YTCOLS, ytCols);
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(ErrDMJ.class);
    job.setJobName(ErrDMJ.class.getSimpleName() + "-" + outPath.getName());
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixAInputPaths, "err");
    
    outPath = fs.makeQualified(outPath);

    for (Path path : matrixAInputPaths) {
      path = fs.makeQualified(path);
      MultipleInputs.addInputPath(job, path, SequenceFileInputFormat.class);
    }

    FileOutputFormat.setOutputPath(job, outPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setNumReduceTasks(0);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed! ");
    return job;
  }

  /**
   * Iterate over the input vectors, do an outer join with the corresponding
   * vector from the other matrix, and write the resulting partial matrix to the
   * reducers.
   * 
   * @author myabandeh
   * 
   */
  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private MapDir xMapDir;
    private SparseMatrix ytMatrix;
    private VectorWritable xVectorw = new VectorWritable();
    double totalDiff = 0;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path mapDirMatrixXPath = new Path(conf.get(MAPDIRMATRIXX));
      xMapDir = new MapDir(conf, mapDirMatrixXPath);
      Path mapDirMatrixYtPath = new Path(conf.get(MAPDIRMATRIXYT));
      int ytRows = conf.getInt(YTROWS, 0);
      int ytCols = conf.getInt(YTCOLS, 0);
      ytMatrix = AlgebraCommon.mapDirToSparseMatrix(mapDirMatrixYtPath, ytRows, ytCols, conf);
    }

    @Override
    public void map(IntWritable index, VectorWritable avw, Context context)
        throws IOException, InterruptedException {
//      System.out.println("Process index " + index);
      Vector av = avw.get();
      Writable xvw = xMapDir.get(index, xVectorw);
      if (xvw == null) {
        // too many nulls could indicate a bug, good to check
        context.getCounter("MapDir", "nullValues").increment(1);
        return;
      }
      Vector xv = xVectorw.get();
      Iterator<Vector.Element> xIt = xv.nonZeroes().iterator();
      while (xIt.hasNext()) {
        Vector.Element xelement = xIt.next();
        int colIndex = xelement.index();
        //row of yt is col of y
        double dotRes = av.dot(ytMatrix.viewRow(colIndex));
        double diff = Math.abs(dotRes - xelement.get());
//        System.out.println("colIndex: " + colIndex + " diff: " + diff
//            + " dotRes: " + dotRes + " av: " + av.zSum() + " yt: "
//            + ytMatrix.viewRow(colIndex).zSum());
        totalDiff += diff;
      }
    }
    
    @Override
    public void cleanup(Context context) throws IOException {
      xMapDir.close();
      int microDiff = (int) (totalDiff * 1000 * 1000);
      System.out.println("totalDiff " + totalDiff + " microDiff " + microDiff);
      context.getCounter("Result", "sumAbsMicro").increment(microDiff);
      context.getCounter("Result", "sumAbsMilli").increment((int) (totalDiff * 1000));
      context.getCounter("Result", "sumAbs").increment((int) (totalDiff));
    }
  }

  @Override
  public int run(String[] args) throws Exception {
    throw new Exception("Not implemented yet");
  }
}
