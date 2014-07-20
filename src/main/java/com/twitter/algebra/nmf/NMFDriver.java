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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.AlgebraCommon;
import com.twitter.algebra.TransposeJob;
import com.twitter.algebra.matrix.format.Sequence2MatrixFormatJob;
import com.twitter.algebra.matrix.multiply.AtB_DMJ;

/**
 * Run NMF (Non-negative Matrix Factorization) on top of MapReduce.
 * 
 * The implementation can also run Sparse NMF.
 * 
 * @author myabandeh
 * 
 */
public class NMFDriver extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(NMFDriver.class);

  private static final String ROWSOPTION = "N";
  private static final String COLSOPTION = "D";
  private static final String PRINCIPALSOPTION = "d";
  private static final String PARTITIONSOPTION = "kp";

  // parameters that control sparse NMF
  private static final String ALPHA1 = "alpha1";
  private static final String ALPHA2 = "alpha2";
  private static final String LAMBDA1 = "lambda1";
  private static final String LAMBDA2 = "lambda2";
  private static final String SAMPLE_RATE = "sampleRate";
  private static final String MIN_ERROR_CHANGE_STR = "matrix.nmf.stop.errchange";
  private static final String MAX_ROUNDS_STR = "matrix.nmf.stop.rounds";
  private float alpha1, alpha2, lambda1, lambda2, sampleRate;
  /**
   * stop the loop if the change in err is less than this
   */
  private long MIN_ERROR_CHANGE;
  private int MAX_ROUNDS;

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(ROWSOPTION, "rows", "Number of rows");
    addOption(COLSOPTION, "cols", "Number of cols");
    addOption(PRINCIPALSOPTION, "pcs", "Number of principal components");
    addOption(PARTITIONSOPTION, "parts", "Number of partitions in principal components");
    addOption(SAMPLE_RATE, SAMPLE_RATE, "sample rate for error calculation");
    if (parseArguments(args) == null) {
      return -1;
    }
    Path input = getInputPath();
    Path output = getOutputPath();
    final int nRows = Integer.parseInt(getOption(ROWSOPTION));
    final int nCols = Integer.parseInt(getOption(COLSOPTION));
    final int nPCs = Integer.parseInt(getOption(PRINCIPALSOPTION));
    final int nColPartitions = Integer.parseInt(getOption(PARTITIONSOPTION));

    alpha1 = Float.parseFloat(getOption(ALPHA1, "0.01"));
    alpha2 = Float.parseFloat(getOption(ALPHA2, "1"));
    lambda1 = Float.parseFloat(getOption(LAMBDA1, "0.01"));
    lambda2 = Float.parseFloat(getOption(LAMBDA2, "0"));

    sampleRate = Float.parseFloat(getOption(SAMPLE_RATE, "0.0001f"));

    Configuration conf = getConf();
    if (conf == null) {
      throw new IOException("No Hadoop configuration present");
    }
    MIN_ERROR_CHANGE = conf.getLong(MIN_ERROR_CHANGE_STR, Long.MAX_VALUE);
    MAX_ROUNDS = conf.getInt(MAX_ROUNDS_STR, 100);
    
    run(conf, input, output, nRows, nCols, nPCs, nColPartitions);
    return 0;
  }

  DistributedRowMatrix threadOutMatrix;
  Thread otherThread;
  
  /**
   * Reads X and factorize it to X = A * Y, where X=r*c, A=r*k, Y=k*c
   * 
   * @param conf
   * @param input path to X in SequenceFileFormat
   * @param output
   * @param nRows
   * @param nCols
   * @param k
   * @param nColPartitionsB
   * @throws Exception
   */
  private void run(final Configuration conf, Path input, Path output, int nRows,
      int nCols, int k, int nColPartitionsB) throws Exception {
    log.info("reading X");
    final DistributedRowMatrix distXOrig =
        new DistributedRowMatrix(input, getTempPath(), nRows, nCols);
    distXOrig.setConf(conf);

    log.info("converting X");
    final DistributedRowMatrix distX = Sequence2MatrixFormatJob.run(conf, distXOrig, "X");

    log.info("writing Xt");
    final DistributedRowMatrix distXt = TransposeJob.transpose(distX, conf, "Xt");

    log.info("summing Xt rows (X cols)");
    Path xtRowSumPath = RowSquareSumJob.run(conf, distXt, "Xt-rowsum");
    Path xColSumPath = xtRowSumPath;
    Vector xColSumVec = AlgebraCommon.mapDirToSparseVector(xColSumPath, 1, 
        distX.numCols(), conf);

    log.info("sampling X");
    DistributedRowMatrix distXr =
        SampleColsJob.run(conf, distX, sampleRate, "Xr");
    log.info("writing Yt");
    DistributedRowMatrix distYt =
        DistRndMatrixJob.random(conf, nCols, k, getTempPath(), "Yt");

    log.info("writing A");
    DistributedRowMatrix distA =
        DistRndMatrixJob.random(conf, nRows, k, getTempPath(), "A");

    long errorChange = -1;
    long prevError = -1;
    //in early rounds sometimes error increases. we should not terminate after that
    for (int round = 0; 
        round < MAX_ROUNDS
        && (errorChange < 0 || errorChange < MIN_ERROR_CHANGE); 
        round++) {
      System.out.println("ROUND " + round + " ......");

      //run in parallel
      final int final_round1 = round;
      final DistributedRowMatrix finalYt = distYt;
      Runnable r = new Runnable() {
        @Override
        public void run() {
          threadOutMatrix = null;
          try {
            threadOutMatrix = //no need to rerun Xt-col (use the same label)
                AtB_DMJ.smartRun(conf, distXt, finalYt, "XYt" + final_round1, "Xt-col", "Yt-col" + final_round1);
          } catch (Exception e) {
            e.printStackTrace();
          }
        }
      };
      otherThread = new Thread(r);
      otherThread.start();
      
      DistributedRowMatrix distYYt =
          new XtXJob().computeXtX(distYt, getTempPath(), conf, "YYt" + round);
      distYYt = CombinerJob.run(conf, distYYt, "YYt-compact" + round);
      
      otherThread.join();
      DistributedRowMatrix distXYt = threadOutMatrix;

      DistributedRowMatrix distAdotXYtdivAYYt =
          CompositeDMJ.run(conf, distA, distXYt, distYYt, "A.XYtdAYYt" + round,
              alpha1, alpha2);
      distA = distAdotXYtdivAYYt;

      //run in parallel
      final int final_round2 = round;
      final DistributedRowMatrix finalA = distA;
      r = new Runnable() {
        @Override
        public void run() {
          threadOutMatrix = null;
          try {
            threadOutMatrix = //no need to rerun Xt-col (use the same label)
                AtB_DMJ.smartRun(conf, distX, finalA, "XtA" + final_round2, "X-col", "A-col" + final_round2);
          } catch (Exception e) {
            e.printStackTrace();
          }
        }
      };
      otherThread = new Thread(r);
      otherThread.start();
      
      DistributedRowMatrix distAAt =
          new XtXJob().computeXtX(distA, getTempPath(), conf, "AAt" + round);
      Matrix centAAt = AlgebraCommon.toDenseMatrix(distAAt);
      // TODO: AtA could be simply a distributed job
      Matrix centAtA = centAAt.transpose();
      DistributedRowMatrix distAtA =
          AlgebraCommon.toMapDir(centAtA, getTempPath(), getTempPath(), "AtA"
              + round);
      
      otherThread.join();
      DistributedRowMatrix distXtA = threadOutMatrix;
      
      DistributedRowMatrix distYtdotXtAdivYtAtA =
          CompositeDMJ.run(conf, distYt, distXtA, distAtA, "Yt.XtAdYtAtA"
              + round, lambda1, lambda2);
      distYt = distYtdotXtAdivYtAtA;

      DistributedRowMatrix distYtr =
          SampleRowsJob.run(conf, distYt, sampleRate, "Ytr" + round);
      distYtr = CombinerJob.run(conf, distYtr, "Ytr-compact" + round);
      long error = ErrDMJ.run(conf, distXr, xColSumVec, distA, distYtr, "ErrJob" + round);
      if (error != -1 && prevError != -1)
        errorChange = (prevError - error);
      prevError = error;
    }
  }

  /**
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new NMFDriver(), args);
  }

}
