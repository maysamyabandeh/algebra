package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.AlgebraCommon;
import com.myabandeh.algebra.TransposeJob;
import com.myabandeh.algebra.matrix.format.Sequence2MatrixFormatJob;
import com.myabandeh.algebra.matrix.multiply.AtBOuterDynamicMapsideJoin;

/**
 * 
 * 
 * @author myabandeh
 * 
 */
public class NMFDriver extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(NMFDriver.class);

  private static final String ROWSOPTION = "N";
  private static final String COLSOPTION = "D";
  private static final String PRINCIPALSOPTION = "d";
  private static final String SPLITFACTOROPTION = "sf";

  private static final String ALPHA1 = "alpha1";
  private static final String ALPHA2 = "alpha2";
  private static final String LAMBDA1 = "lambda1";
  private static final String LAMBDA2 = "lambda2";
  /**
   * We use a single random object to help reproducing the erroneous scenarios
   */
  static Random random = new Random(0);

  private float alpha1, alpha2, lambda1, lambda2;
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(ROWSOPTION, "rows", "Number of rows");
    addOption(COLSOPTION, "cols", "Number of cols");
    addOption(PRINCIPALSOPTION, "pcs", "Number of principal components");
    addOption(SPLITFACTOROPTION, "sf",
        "Split each block to increase paralelism");
    if (parseArguments(args) == null) {
      return -1;
    }
    Path input = getInputPath();
    Path output = getOutputPath();
    final int nRows = Integer.parseInt(getOption(ROWSOPTION));
    final int nCols = Integer.parseInt(getOption(COLSOPTION));
    final int nPCs = Integer.parseInt(getOption(PRINCIPALSOPTION));
    final int splitFactor = Integer.parseInt(getOption(SPLITFACTOROPTION, "1"));
    
    alpha1 = Float.parseFloat(getOption(ALPHA1, "0.01"));
    alpha2 = Float.parseFloat(getOption(ALPHA2, "1"));
    lambda1 = Float.parseFloat(getOption(LAMBDA1, "0.01"));
    lambda2 = Float.parseFloat(getOption(LAMBDA2, "0"));

    Configuration conf = getConf();
    if (conf == null) {
      throw new IOException("No Hadoop configuration present");
    }
    run(conf, input, output, nRows, nCols, nPCs, splitFactor);
    return 0;
  }

  final static float SAMPLE_RATE = 0.0001f;

  /**
   * Reads input matrix Y and generates A,A',B,C,C'. Then runs the benchmarks.
   * 
   * @param conf
   * @param input
   * @param output
   * @param nRows
   * @param nCols
   * @param k
   * @param splitFactor
   * @throws Exception
   */
  private void run(Configuration conf, Path input, Path output, int nRows,
      int nCols, int k, int splitFactor) throws Exception {
//    FileSystem fs = FileSystem.get(input.toUri(), conf);
//    int blockSize = (int) fs.getFileStatus(input).getBlockSize();
    //init
    log.info("reading X");
    DistributedRowMatrix distX =
        new DistributedRowMatrix(input, getTempPath(), nRows, nCols);
    distX.setConf(conf);

    log.info("converting X");
    distX = Sequence2MatrixFormatJob.run(conf, distX, "X");
    
    log.info("writing Xt");
    DistributedRowMatrix distXt =
        TransposeJob.transpose(distX, conf, "Xt");

    log.info("sampling X");
    DistributedRowMatrix distXr = SampleColsJob.run(conf, distX, SAMPLE_RATE, "Xr");
    log.info("writing Yt");
    // Matrix centYt = randomMatrix(nCols, k);
    // DistributedRowMatrix distYt =
    // AlgebraCommon.toMapDir(centYt, getTempPath(), getTempPath(), "Yt");
    DistributedRowMatrix distYt =
        DistRndMatrixJob.random(conf, nCols, k, getTempPath(),"Yt");
    
    final int NUM_COL_PARTITIONS = 500;
    final int COLPARTITION_SIZE = ColPartitionJob.getColPartitionSize(k, NUM_COL_PARTITIONS);

    log.info("writing A");
    // Matrix centA = randomMatrix(nRows, k);
    // DistributedRowMatrix distA =
    // AlgebraCommon.toMapDir(centA, getTempPath(), getTempPath(), "A");
    DistributedRowMatrix distA =
        DistRndMatrixJob.random(conf, nRows, k, getTempPath(), "A");

    for (int round = 0; round < 100; round ++) {

    DistributedRowMatrix distYYt = new XtXJob().computeXtX(distYt, getTempPath(), conf, "YYt" + round);
    distYYt = CombinerJob.run(conf, distYYt, "YYt-compact" + round);
    DistributedRowMatrix distYtCol = ColPartitionJob.partition(distYt, conf, "Ytcol" + round, NUM_COL_PARTITIONS);
    DistributedRowMatrix distXYt = DMJ.run(conf, distXt, distYtCol, COLPARTITION_SIZE, "XYt" + round, false, 1);

    DistributedRowMatrix distAdotXYtdivAYYt = 
        CompositeDMJ.run(conf, distA, distXYt, distYYt, "A.XYtdAYYt" + round, alpha1, alpha2, 1);
    distA = distAdotXYtdivAYYt;

    DistributedRowMatrix distAAt = new XtXJob().computeXtX(distA, getTempPath(), conf, "AAt" + round);
//    distAAt = CombinerJob.run(conf, distAAt, "AAt-compact" + round);
    Matrix centAAt = AlgebraCommon.toDenseMatrix(distAAt);
    //TODO: AtA could be simply a distributed job
    Matrix centAtA = centAAt.transpose();
    DistributedRowMatrix distAtA = AlgebraCommon.toMapDir(centAtA, getTempPath(), getTempPath(), "AtA" + round);
    DistributedRowMatrix distACol = ColPartitionJob.partition(distA, conf, "Acol" + round, NUM_COL_PARTITIONS);
    DistributedRowMatrix distXtA = DMJ.run(conf, distX, distACol, COLPARTITION_SIZE, "XtA" + round, false, 1);
//    DistributedRowMatrix distXtA = AtBOuterDynamicMapsideJoin.run(conf, distX, distA, "XtA" + round, true, 1);
    DistributedRowMatrix distYtdotXtAdivYtAtA = 
        CompositeDMJ.run(conf, distYt, distXtA, distAtA, "Yt.XtAdYtAtA" + round, lambda1, lambda2, 1);
    distYt = distYtdotXtAdivYtAtA;

    DistributedRowMatrix distYtr = SampleRowsJob.run(conf, distYt, SAMPLE_RATE, "Ytr" + round);
    distYtr = CombinerJob.run(conf, distYtr, "Ytr-compact" + round);
    ErrDMJ.run(conf, distXr, distA, distYtr, "ErrJob" + round);
    
    }

  }


  /**
   * A randomly initialized matrix
   * 
   * @param rows
   * @param cols
   * @return
   */
  static Matrix randomMatrix(int rows, int cols) {
    Matrix randM = new DenseMatrix(rows, cols);
    randM.assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return random.nextDouble();
      }
    });
    return randM;
  }

  /**
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new NMFDriver(), args);
  }

}
