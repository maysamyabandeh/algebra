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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.AlgebraCommon;
import com.twitter.algebra.MergeVectorsReducer;
import com.twitter.algebra.matrix.format.MapDir;
import com.twitter.algebra.matrix.format.MatrixOutputFormat;

/**
 * | X - A * Y |
 * @author myabandeh
 */
public class ErrDMJ extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(ErrDMJ.class);

  public static final String MAPDIRMATRIXX = "mapDirMatrixX";
  public static final String MAPDIRMATRIXYT = "mapDirMatrixYt";
  public static final String YTROWS = "YtRows";
  public static final String YTCOLS = "YtCols";

  public static long run(Configuration conf, DistributedRowMatrix X, Vector xColSumVec,
      DistributedRowMatrix A, DistributedRowMatrix Yt, String label)
      throws IOException, InterruptedException, ClassNotFoundException {
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
    long totalErr = -1;
    if (!fs.exists(outPath)) {
      Job hJob = job.run(conf, X.getRowPath(), A.getRowPath(),
          Yt.getRowPath(), outPath, A.numRows(), Yt.numRows(), Yt.numCols());
      Counters counters = hJob.getCounters();
      counters.findCounter("Result", "sumAbs").getValue();
      log.info("FINAL ERR is " + totalErr);
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    Vector sumErrVec = AlgebraCommon.mapDirToSparseVector(outPath, 1, 
        X.numCols(), conf);
    double maxColErr = Double.MIN_VALUE;
    double sumColErr = 0;
    int cntColErr = 0;
    Iterator<Vector.Element> it = sumErrVec.nonZeroes().iterator();
    while (it.hasNext()) {
      Vector.Element el = it.next();
      double errP2 = el.get();
      double origP2 =  xColSumVec.get(el.index());
      double colErr = Math.sqrt(errP2 / origP2);
      log.info("col: " + el.index() + " sum(err^2): " + errP2 + " sum(val^2): " + 
          origP2 + " colErr: " + colErr);
      maxColErr = Math.max(colErr, maxColErr);
      sumColErr += colErr;
      cntColErr++;
    }
    log.info(" Max Col Err: " + maxColErr);
    log.info(" Avg Col Err: " + sumColErr / cntColErr);
    return totalErr;
  }

  public Job run(Configuration conf, Path xPath, Path matrixAInputPath,
      Path ytPath, Path outPath, int aRows, int ytRows, int ytCols)
      throws IOException, InterruptedException, ClassNotFoundException {
    conf = new Configuration(conf);

    conf.set(MAPDIRMATRIXX, xPath.toString());
    conf.set(MAPDIRMATRIXYT, ytPath.toString());
    conf.setInt(YTROWS, ytRows);
    conf.setInt(YTCOLS, ytCols);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixAInputPath, "err");

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(ErrDMJ.class);
    job.setJobName(ErrDMJ.class.getSimpleName() + "-" + outPath.getName());

    matrixAInputPath = fs.makeQualified(matrixAInputPath);
    MultipleInputs.addInputPath(job, matrixAInputPath,
        SequenceFileInputFormat.class);

    outPath = fs.makeQualified(outPath);
    FileOutputFormat.setOutputPath(job, outPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    int numReducers = 1;
    job.setNumReduceTasks(numReducers);
    job.setCombinerClass(MergeVectorsReducer.class);
    job.setReducerClass(MergeVectorsReducer.class);
    
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed! ");
    return job;
  }

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private MapDir xMapDir;
    private SparseMatrix ytMatrix;
    private VectorWritable xVectorw = new VectorWritable();
    private VectorWritable outvw = new VectorWritable();
    private IntWritable iw = new IntWritable(0);
    double totalDiff = 0;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path mapDirMatrixXPath = new Path(conf.get(MAPDIRMATRIXX));
      xMapDir = new MapDir(conf, mapDirMatrixXPath);
      Path mapDirMatrixYtPath = new Path(conf.get(MAPDIRMATRIXYT));
      int ytRows = conf.getInt(YTROWS, 0);
      int ytCols = conf.getInt(YTCOLS, 0);
      ytMatrix =
          AlgebraCommon.mapDirToSparseMatrix(mapDirMatrixYtPath, ytRows,
              ytCols, conf);
    }

    @Override
    public void map(IntWritable index, VectorWritable avw, Context context)
        throws IOException, InterruptedException {
      Vector av = avw.get();
      Writable xvw = xMapDir.get(index, xVectorw);
      if (xvw == null) {
        // too many nulls could indicate a bug, good to check
        context.getCounter("MapDir", "nullValues").increment(1);
        return;
      }
      Vector xv = xVectorw.get();
      Vector sumVec = new RandomAccessSparseVector(xv.size());
      Iterator<Vector.Element> xIt = xv.nonZeroes().iterator();
      while (xIt.hasNext()) {
        Vector.Element xelement = xIt.next();
        int colIndex = xelement.index();
        // row of yt is col of y
        double dotRes = av.dot(ytMatrix.viewRow(colIndex));
        double diff = dotRes - xelement.get();
        totalDiff += Math.abs(diff);
        double squareDiff = diff * diff;
        sumVec.set(colIndex, squareDiff);
        outvw.set(sumVec);
        context.write(iw, outvw);
      }
    }

    @Override
    public void cleanup(Context context) throws IOException {
      xMapDir.close();
      int microDiff = (int) (totalDiff * 1000 * 1000);
      System.out.println("totalDiff " + totalDiff + " microDiff " + microDiff);
      context.getCounter("Result", "sumAbsMicro").increment(microDiff);
      context.getCounter("Result", "sumAbsMilli").increment(
          (int) (totalDiff * 1000));
      context.getCounter("Result", "sumAbs").increment((int) (totalDiff));
    }
  }

  @Override
  public int run(String[] args) throws Exception {
    throw new Exception("Not implemented yet");
  }
}
