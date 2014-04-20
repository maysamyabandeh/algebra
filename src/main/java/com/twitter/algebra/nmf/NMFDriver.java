package com.twitter.algebra.nmf;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.AlgebraCommon;
import com.myabandeh.algebra.TransposeJob;
import com.myabandeh.algebra.matrix.format.Sequence2MatrixFormatJob;
import com.myabandeh.algebra.matrix.multiply.AtB_DMJ;

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
  private float alpha1, alpha2, lambda1, lambda2, sampleRate;

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(ROWSOPTION, "rows", "Number of rows");
    addOption(COLSOPTION, "cols", "Number of cols");
    addOption(PRINCIPALSOPTION, "pcs", "Number of principal components");
    addOption(PARTITIONSOPTION, "parts", "Number of partitions in principal components");
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
    run(conf, input, output, nRows, nCols, nPCs, nColPartitions);
    return 0;
  }

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
  private void run(Configuration conf, Path input, Path output, int nRows,
      int nCols, int k, int nColPartitionsB) throws Exception {
    log.info("reading X");
    DistributedRowMatrix distX =
        new DistributedRowMatrix(input, getTempPath(), nRows, nCols);
    distX.setConf(conf);

    log.info("converting X");
    distX = Sequence2MatrixFormatJob.run(conf, distX, "X");

    log.info("writing Xt");
    DistributedRowMatrix distXt = TransposeJob.transpose(distX, conf, "Xt");

    log.info("sampling X");
    DistributedRowMatrix distXr =
        SampleColsJob.run(conf, distX, sampleRate, "Xr");
    log.info("writing Yt");
    DistributedRowMatrix distYt =
        DistRndMatrixJob.random(conf, nCols, k, getTempPath(), "Yt");

    log.info("writing A");
    DistributedRowMatrix distA =
        DistRndMatrixJob.random(conf, nRows, k, getTempPath(), "A");

    for (int round = 0; round < 100; round++) {
      System.out.println("ROUND " + round + " ......");

      DistributedRowMatrix distYYt =
          new XtXJob().computeXtX(distYt, getTempPath(), conf, "YYt" + round);
      distYYt = CombinerJob.run(conf, distYYt, "YYt-compact" + round);
//      DistributedRowMatrix distYtCol =
//          ColPartitionJob.partition(distYt, conf, "Ytcol" + round,
//              nColPartitionsB);
//      DistributedRowMatrix distXYt =
//          AtB_DMJ.run(conf, distXt, distYtCol, 1,
//              nColPartitionsB, "XYt" + round, true);
      DistributedRowMatrix distXYt =
          AtB_DMJ.smartRun(conf, distXt, distYt, "XYt" + round);

      DistributedRowMatrix distAdotXYtdivAYYt =
          CompositeDMJ.run(conf, distA, distXYt, distYYt, "A.XYtdAYYt" + round,
              alpha1, alpha2);
      distA = distAdotXYtdivAYYt;

      DistributedRowMatrix distAAt =
          new XtXJob().computeXtX(distA, getTempPath(), conf, "AAt" + round);
      Matrix centAAt = AlgebraCommon.toDenseMatrix(distAAt);
      // TODO: AtA could be simply a distributed job
      Matrix centAtA = centAAt.transpose();
      DistributedRowMatrix distAtA =
          AlgebraCommon.toMapDir(centAtA, getTempPath(), getTempPath(), "AtA"
              + round);
//      DistributedRowMatrix distACol =
//          ColPartitionJob.partition(distA, conf, "Acol" + round,
//              nColPartitionsB);
//      DistributedRowMatrix distXtA =
//          AtB_DMJ.run(conf, distX, distACol, 1, nColPartitionsB,
//              "XtA" + round, true);
      DistributedRowMatrix distXtA =
          AtB_DMJ.smartRun(conf, distX, distA, "XtA" + round);
      DistributedRowMatrix distYtdotXtAdivYtAtA =
          CompositeDMJ.run(conf, distYt, distXtA, distAtA, "Yt.XtAdYtAtA"
              + round, lambda1, lambda2);
      distYt = distYtdotXtAdivYtAtA;

      DistributedRowMatrix distYtr =
          SampleRowsJob.run(conf, distYt, sampleRate, "Ytr" + round);
      distYtr = CombinerJob.run(conf, distYtr, "Ytr-compact" + round);
      ErrDMJ.run(conf, distXr, distA, distYtr, "ErrJob" + round);
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
